/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file calibrate.cc
 *
 * \brief Create profile graph and calibrate on dataset
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>

#include <numeric>

#include "./quantize.h"

namespace tvm {
namespace relay {
namespace quantize {

// KL divergence minimization code is adapted from MXNet.
// The original one is in incubator-mxnet/src/operator/quantization/calibrate.cc
static std::vector<float> SmoothDistribution(const std::vector<float>& p,
                                             const float eps = 0.0001) {
  std::vector<size_t> is_zeros(p.size());
  std::vector<size_t> is_nonzeros(p.size());
  {
    auto it = p.begin();
    std::generate(is_zeros.begin(), is_zeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) == 0.f); });
  }
  {
    auto it = p.begin();
    std::generate(is_nonzeros.begin(), is_nonzeros.end(),
                  [&it]() { return static_cast<size_t>(*(it++) != 0.f); });
  }
  size_t n_zeros = std::accumulate(is_zeros.begin(), is_zeros.end(), 0);
  size_t n_nonzeros = p.size() - n_zeros;
  if (!n_nonzeros) {
    // The discrete probability distribution is malformed. All entries are 0.
    return std::vector<float>();
  }
  float eps1 = eps * static_cast<float>(n_zeros) / static_cast<float>(n_nonzeros);
  if (eps1 >= 1.0) return std::vector<float>();
  auto ret = p;
  for (size_t i = 0; i < p.size(); i++) {
    ret[i] += eps * is_zeros[i] - eps1 * is_nonzeros[i];
  }
  return ret;
}

static float ComputeEntropy(float* p, float* q, size_t size) {
  float p_sum = std::accumulate(p, p + size, 0.f);
  float q_sum = std::accumulate(q, q + size, 0.f);
  float ret = 0;
  for (size_t i = 0; i < size; i++) {
    ICHECK(p[i] > 0 && q[i] > 0);
    p[i] /= p_sum;
    q[i] /= q_sum;
    if (p[i] && q[i]) ret += p[i] * std::log(p[i] / q[i]);
  }
  return ret;
}

void MinimizeKL(const std::vector<int>& hist, const std::vector<float>& hist_edges, 
                 float* divergence_out, int num_bins, int num_quantized_bins) {
  const int zero_bin_idx = num_bins / 2;
  const int num_half_quantized_bins = num_quantized_bins / 2;
  std::vector<float> thresholds(num_bins / 2 + 1 - num_quantized_bins / 2, 0.f);
  std::vector<float> quantized_bins(num_quantized_bins, 0);
  for (int i = num_quantized_bins / 2; i < zero_bin_idx + 1; ++i) {
    const int p_bin_idx_start = zero_bin_idx - i;
    const int p_bin_idx_stop = zero_bin_idx + i + 1;
    thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop];

    std::vector<int> sliced_nd_hist(p_bin_idx_stop - p_bin_idx_start);
    std::vector<float> p(sliced_nd_hist.size());
    p[0] = 0;
    p.back() = 0;
    for (int j = 0; j < num_bins; j++) {
      if (j <= p_bin_idx_start) {
        p[0] += hist[j];
      } else if (j >= p_bin_idx_stop) {
        p.back() += hist[j];
      } else {
        sliced_nd_hist[j - p_bin_idx_start] = hist[j];
        p[j - p_bin_idx_start] = hist[j];
      }
    }
    // calculate how many bins should be merged to generate quantized distribution q
    const auto num_merged_bins = sliced_nd_hist.size() / num_quantized_bins;
    for (int j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j + 1) * num_merged_bins;
      quantized_bins[j] =
          std::accumulate(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop, 0);
    }
    quantized_bins.back() += std::accumulate(
        sliced_nd_hist.begin() + static_cast<int>(num_quantized_bins * num_merged_bins),
        sliced_nd_hist.end(), 0);
    // expand quantized_bins into p.size bins
    std::vector<float> q(sliced_nd_hist.size(), 0);
    for (int j = 0; j < num_quantized_bins; j++) {
      const int start = j * num_merged_bins;
      const int stop = (j == num_quantized_bins - 1) ? q.size() : ((j + 1) * num_merged_bins);
      int norm = std::count_if(sliced_nd_hist.begin() + start, sliced_nd_hist.begin() + stop,
                               [](size_t i) { return i != 0; });
      if (norm) {
        for (int k = start; k < stop; k++) {
          if (p[k]) q[k] = quantized_bins[j] / norm;
        }
      }
    }
    p = SmoothDistribution(p);
    q = SmoothDistribution(q);

    if (!q.size()) {
      divergence_out[i - num_half_quantized_bins] += std::numeric_limits<float>::infinity();
    } else {
      divergence_out[i - num_half_quantized_bins] += ComputeEntropy(p.data(), q.data(), p.size());
    }
  }
}

class StatsCollector : private ExprMutator {
 public:
  StatsCollector() : simulated_quantize_op_(Op::Get("relay.op.annotation.simulated_quantize")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& simulated_quantize_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == simulated_quantize_op_) {
      auto attrs = new_call->attrs.as<SimulatedQuantizeAttrs>();
      // rewrite the annotation
      auto new_attrs = make_object<SimulatedQuantizeAttrs>();
      const Expr& quantize_input = new_call->args[0];                  // expression being quantized
      auto placeholder_scale = MakeConstantTensor(DataType::Float(32), std::vector<int64_t>(1, 1), std::vector<int64_t>(1, 1));  // unused argument
      auto placeholder_clip = MakeConstantScalar(DataType::Float(32), 0.);
      auto placeholder_zero = MakeConstantScalar(DataType::Float(32), 0.);
      auto placeholder_zero_1 = MakeConstantTensor(DataType::Float(32), std::vector<int64_t>(1, 1), std::vector<int64_t>(1, 1));
      //auto placeholder = MakeConstantTensor(DataType::Float(32), 0.);
      Array<Expr> new_args{quantize_input, placeholder_scale, placeholder_clip, placeholder_clip, placeholder_zero};
      if (attrs->asymmetric)
        new_args = {quantize_input, placeholder_scale, placeholder_clip, placeholder_clip, placeholder_zero_1};

      new_attrs->kind = QAnnotateKind::kQIdentity;
      new_attrs->rounding = attrs->rounding;
      new_attrs->per_channel = attrs->per_channel;
      new_attrs->asymmetric = attrs->asymmetric;
      new_attrs->name = attrs->name;
      Expr identity_quantize = Call(new_call->op, new_args, Attrs{new_attrs}, {});

      // add non-const expressions to profile data
      if (attrs->kind != QAnnotateKind::kQWeight && attrs->kind != QAnnotateKind::kQBias) {
        ICHECK(!quantize_input.as<ConstantNode>());
        profile_data_.push_back(identity_quantize);
      }
      return identity_quantize;
    } else {
      return new_e;
    }
  }
};

class QWeightCollector : private ExprMutator {
 public:
  QWeightCollector() : simulated_quantize_op_(Op::Get("relay.op.annotation.simulated_quantize")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& simulated_quantize_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == simulated_quantize_op_) {
      auto attrs = new_call->attrs.as<SimulatedQuantizeAttrs>();
      const Expr& quantize_input = new_call->args[0];                  // expression being quantized
      // add non-const expressions to profile data
      if (attrs->kind == QAnnotateKind::kQWeight || attrs->kind == QAnnotateKind::kQBias) {
        ICHECK(quantize_input.as<ConstantNode>());
        profile_data_.push_back(new_e);
      }
      return new_e;
    } else {
      return new_e;
    }
  }
};

class QActCollector : private ExprMutator {
 public:
  QActCollector() : simulated_quantize_op_(Op::Get("relay.op.annotation.simulated_quantize")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& simulated_quantize_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == simulated_quantize_op_) {
      auto attrs = new_call->attrs.as<SimulatedQuantizeAttrs>();
      const Expr& quantize_input = new_call->args[0];                  // expression being quantized
      // add non-const expressions to profile data
      if (attrs->kind != QAnnotateKind::kQWeight && attrs->kind != QAnnotateKind::kQBias) {
        ICHECK(!quantize_input.as<ConstantNode>());
        profile_data_.push_back(new_e);
      }
      return new_e;
    } else {
      return new_e;
    }
  }
};

class QActAllCollector : private ExprMutator {
 public:
  QActAllCollector() : simulated_quantize_op_(Op::Get("relay.op.annotation.simulated_quantize")), 
                       cast_hint_op_(Op::Get("annotation.cast_hint")),
                       stop_fusion_op_(Op::Get("annotation.stop_fusion")),
                       split_op_(Op::Get("split")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& simulated_quantize_op_;
  const Op& cast_hint_op_;
  const Op& stop_fusion_op_;
  const Op& split_op_;
  int count = 0;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op != simulated_quantize_op_ && new_call->op != cast_hint_op_ && 
        new_call->op != stop_fusion_op_ && new_call->op != split_op_) {
      const Expr& quantize_input = new_call->args[0];                  // expression being quantized
      // add non-const expressions to profile data
      ICHECK(!quantize_input.as<ConstantNode>());
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};


////////////ztl

class ReluCollector : private ExprMutator {
 public:
  ReluCollector() : relu_op_(Op::Get("nn.relu")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& relu_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == relu_op_) {
      const Expr& quantize_input = new_call->args[0];                  // expression being quantized
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};


class QCheckPointCollector : private ExprMutator {
 public:
  QCheckPointCollector() : check_point_op_(Op::Get("annotation.checkpoint")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& check_point_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    Op op = Downcast<Op>(new_call->op);
    //printf("this op name is %s.\n", op->name.c_str());
    ICHECK(new_call);
    if (new_call->op == check_point_op_) {
      //printf("add checkpoint collecter\n");
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};

class QCheckPointSiSoCollector : private ExprMutator {
 public:
  QCheckPointSiSoCollector() : check_point_op_(Op::Get("annotation.checkpointsiso")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& check_point_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    Op op = Downcast<Op>(new_call->op);
    //printf("this op name is %s.\n", op->name.c_str());
    ICHECK(new_call);
    if (new_call->op == check_point_op_) {
      //printf("add checkpoint collecter\n");
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};

class QCheckPointBiasSCollector : private ExprMutator {
 public:
  QCheckPointBiasSCollector() : check_point_op_(Op::Get("annotation.checkpointbiass")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& check_point_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    Op op = Downcast<Op>(new_call->op);
    //printf("this op name is %s.\n", op->name.c_str());
    ICHECK(new_call);
    if (new_call->op == check_point_op_) {
      //printf("add checkpoint collecter\n");
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};






class QAddCollector : private ExprMutator {
 public:
  QAddCollector() : add_op_(Op::Get("add")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& add_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    //const CallNode* CallOp = new_call->args[1].as<CallNode>();
    //Op op = Downcast<Op>(new_call->args[1].as<CallNode>()->op);
    ICHECK(new_call);
    //static const Op& simulated_quantize = Op::Get("relay.op.annotation.simulated_quantize");
    //if (new_call->op == add_op_ && op->name != "relay.op.annotation.simulated_quantize") {
    //if (new_call->op == add_op_ && !CallOp->op.same_as(simulated_quantize)) {  
    if (new_call->op == add_op_) {    
    //if (new_call->op == add_op_) {
      //const Expr& quantize_input = new_call->args[0];     
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};

class QConvCollector : private ExprMutator {
 public:
  QConvCollector() : conv_op_(Op::Get("nn.conv2d")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};
class QCheckPointBiasCollector : private ExprMutator {
 public:
  QCheckPointBiasCollector() : conv_op_(Op::Get("annotation.checkpointbias")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};



class QCheckPointAddOutputCollector : private ExprMutator {
 public:
  QCheckPointAddOutputCollector() : conv_op_(Op::Get("annotation.checkpointaddoutput")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};


class QCheckPointWeightCollector : private ExprMutator {
 public:
  QCheckPointWeightCollector() : conv_op_(Op::Get("annotation.checkpointweight")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};


class QCheckPointInputCollector : private ExprMutator {
 public:
  QCheckPointInputCollector() : conv_op_(Op::Get("annotation.checkpointinput")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};



class QCheckPointZpiCollector : private ExprMutator {
 public:
  QCheckPointZpiCollector() : conv_op_(Op::Get("annotation.checkpointzpi")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};


class QCheckPointZpwCollector : private ExprMutator {
 public:
  QCheckPointZpwCollector() : conv_op_(Op::Get("annotation.checkpointzpw")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};


class QCheckPointSiCollector : private ExprMutator {
 public:
  QCheckPointSiCollector() : conv_op_(Op::Get("annotation.checkpointsi")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};

class QCheckPointSwCollector : private ExprMutator {
 public:
  QCheckPointSwCollector() : conv_op_(Op::Get("annotation.checkpointsw")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    ICHECK(func) << "Input shoule be Function";
    Expr new_body = Tuple(std::move(profile_data_));
    Function ret_func = WithFields(GetRef<Function>(func), FreeVars(new_body), new_body);

    // We are changing the function's ret_type to an empty type. Unfortunately, Optional<Type>() is
    // indistinguishable from NullValue<Type>(), so we can't express "update to nullptr" in
    // WithFields.
    ret_func.CopyOnWrite()->ret_type = NullValue<Type>();
    return ret_func;
  }

 private:
  Array<Expr> profile_data_;
  const Op& conv_op_;


  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    ICHECK(new_call);
    if (new_call->op == conv_op_ ) {       
      profile_data_.push_back(new_e);
      return new_e;
    } else {
      return new_e;
    }
  }
};



/*
 * \brief Given an annotated graph, create a profile graph to collect profile data from the
 * calibration dataset.
 *
 * This pass collects simulated_quantize op into a tuple. Simulated_quantize ops are rewritten to
 * identity mode. The tuple is the output of the profile graph. Both input and output of this pass
 * are relay::Function.
 *
 * \param expr The simulation graph after annotation.
 * \return The profile graph.
 */
Expr CreateStatsCollector(const Expr& expr) { return StatsCollector().Collect(expr); }
Expr CreateQWeightCollector(const Expr& expr) { return QWeightCollector().Collect(expr); }
Expr CreateQActCollector(const Expr& expr) { return QActCollector().Collect(expr); }
Expr CreateQActAllCollector(const Expr& expr) { return QActAllCollector().Collect(expr); }
Expr CreateQCheckPointCollector(const Expr& expr) { return QCheckPointCollector().Collect(expr); }
Expr CreateQAddCollector(const Expr& expr) { return QAddCollector().Collect(expr); }
Expr CreateQConvCollector(const Expr& expr) { return QConvCollector().Collect(expr); }
Expr CreateReluCollector(const Expr& expr) { return ReluCollector().Collect(expr); }

Expr CreateQCheckPointSiSoCollector(const Expr& expr) { return QCheckPointSiSoCollector().Collect(expr);}
Expr CreateQCheckPointBiasSCollector(const Expr& expr) { return QCheckPointBiasSCollector().Collect(expr);}
Expr CreateQCheckPointBiasCollector(const Expr& expr) { return QCheckPointBiasCollector().Collect(expr);}
Expr CreateQCheckPointAddOutputCollector(const Expr& expr) { return QCheckPointAddOutputCollector().Collect(expr);}
Expr CreateQCheckPointWeightCollector(const Expr& expr) { return QCheckPointWeightCollector().Collect(expr);}
Expr CreateQCheckPointInputCollector(const Expr& expr) { return QCheckPointInputCollector().Collect(expr);}
Expr CreateQCheckPointZpiCollector(const Expr& expr) { return QCheckPointZpiCollector().Collect(expr);}
Expr CreateQCheckPointZpwCollector(const Expr& expr) { return QCheckPointZpwCollector().Collect(expr);}
Expr CreateQCheckPointSiCollector(const Expr& expr) { return QCheckPointSiCollector().Collect(expr);}
Expr CreateQCheckPointSwCollector(const Expr& expr) { return QCheckPointSwCollector().Collect(expr);}

TVM_REGISTER_GLOBAL("relay._quantize.CreateStatsCollector").set_body_typed(CreateStatsCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQWeightCollector").set_body_typed(CreateQWeightCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQActCollector").set_body_typed(CreateQActCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQActAllCollector").set_body_typed(CreateQActAllCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointCollector").set_body_typed(CreateQCheckPointCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQAddCollector").set_body_typed(CreateQAddCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQConvCollector").set_body_typed(CreateQConvCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateReluCollector").set_body_typed(CreateReluCollector);


TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointSiSoCollector").set_body_typed(CreateQCheckPointSiSoCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointBiasSCollector").set_body_typed(CreateQCheckPointBiasSCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointBiasCollector").set_body_typed(CreateQCheckPointBiasCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointAddOutputCollector").set_body_typed(CreateQCheckPointAddOutputCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointWeightCollector").set_body_typed(CreateQCheckPointWeightCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointInputCollector").set_body_typed(CreateQCheckPointInputCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointZpiCollector").set_body_typed(CreateQCheckPointZpiCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointZpwCollector").set_body_typed(CreateQCheckPointZpwCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointSiCollector").set_body_typed(CreateQCheckPointSiCollector);
TVM_REGISTER_GLOBAL("relay._quantize.CreateQCheckPointSwCollector").set_body_typed(CreateQCheckPointSwCollector);






TVM_REGISTER_GLOBAL("relay._quantize.FindScaleByKLMinimization")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int* hist_ptr = static_cast<int*>(static_cast<void*>(args[0]));
      float* hist_edges_ptr = static_cast<float*>(static_cast<void*>(args[1]));
      float* divergence_out_ptr = static_cast<float*>(static_cast<void*>(args[2]));
      int num_bins = args[3];
      int num_quantized_bins = args[4];
      std::vector<int> hist(hist_ptr, hist_ptr + num_bins);
      std::vector<float> hist_edges(hist_edges_ptr, hist_edges_ptr + num_bins + 1);
      MinimizeKL(hist, hist_edges, divergence_out_ptr, num_bins, num_quantized_bins);
    });

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
