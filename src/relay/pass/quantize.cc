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
 * Copyright (c) 2018 by Contributors
 *
 * \file quantize.cc
 *
 * \brief transform a graph to a low-bit graph
 *   for compression and acceleration.
 */
#include <dmlc/thread_local.h>
#include <tvm/base.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <cmath>
#include <string>
#include <vector>
#include <stack>
#include <utility>
#include "pattern_util.h"
#include "quantize.h"


namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

/*! \brief Attribute for simulated quantize operator */
struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  int kind;
  bool sign;
  std::string rounding;
  int passthrough;
  std::string granularity;
  std::string layout;
  std::string op_hint;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(kind)
        .describe("kind of field, hint for nbit/dtype configuration.");
    TVM_ATTR_FIELD(sign).set_default(true)
        .describe("whether to use signed data type.");
    TVM_ATTR_FIELD(rounding).set_default("round")
        .describe("rounding mode. Can be 'floor', 'ceil', 'round'");
    TVM_ATTR_FIELD(passthrough).set_default(false)
        .describe("whether to passthrough full precision value (useful for\
                   data-aware calibration)");
    TVM_ATTR_FIELD(granularity).set_default("layer")
        .describe("scale granularity. Can be 'global', 'layer', 'channel'");
    TVM_ATTR_FIELD(layout).set_default("unknown")
        .describe("data layout (e.g., 'NCHW', 'NHWC')");
    TVM_ATTR_FIELD(op_hint).set_default("")
        .describe("operator hint on how to interpret layout (e.g., 'broadcastable')");
  }
};

TVM_REGISTER_NODE_TYPE(SimulatedQuantizeAttrs);

bool SimulatedQuantizeRel(const Array<Type>& types,
                          int num_inputs,
                          const Attrs& attrs,
                          const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto param = attrs.as<SimulatedQuantizeAttrs>();
  CHECK(param != nullptr);

  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  CHECK_NE(data->shape.size(), 0) << "Input shape cannot be empty";

  size_t channel_dim = param->layout.find("C");
  if (channel_dim == std::string::npos)
    channel_dim = param->layout.find("I");

  // TODO(eqy): blocked layouts
  CHECK(param->layout.find_first_not_of("NCOIHW") == std::string::npos) << "Unsupported Layout in Simulated Quantize";

  int channels = 1;
  if (data->shape.size() >= 4)  {
    channels = data->shape[channel_dim].as<IntImm>()->value;
  } else if (param->op_hint.find("broadcastable") != std::string::npos && data->shape.size() == 3) {
    // TODO(eqy): robust broadcast handling, blocked layout support
    size_t d = 0;
    for (; d < data->shape.size(); d++) {
      if (data->shape[d].as<IntImm>()->value != 1) {
        channels = data->shape[d].as<IntImm>()->value;
        break;
      }
    }
    for (d = d + 1;d < data->shape.size(); d++) {
      CHECK_EQ(data->shape[d].as<IntImm>()->value, 1)
      << "Unhandled broadcastable data shape"
      << data->shape;
    }
  } else {
    channels = 1;
  }

  if (param->granularity == "channel") {
    reporter->Assign(types[1], TensorTypeNode::make({channels,}, Float(32)));    // dom_scale
  } else {
    reporter->Assign(types[1], TensorTypeNode::make({1}, Float(32)));    // dom_scale
  }
  reporter->Assign(types[2], TensorTypeNode::make({}, Float(32)));    // clip_min
  reporter->Assign(types[3], TensorTypeNode::make({}, Float(32)));    // clip_max
  reporter->Assign(types[4], types[0]);                               // output
  return true;
}

RELAY_REGISTER_OP("relay.op.annotation.simulated_quantize")
.describe(R"code(simulated quantize op)code" TVM_ADD_FILELINE)
.set_num_inputs(4)
.add_argument("data", "Tensor", "The input data.")
.add_argument("dom_scale", "Tensor", "The domain scale of input data. It should be a scalar")
.add_argument("clip_min", "Tensor", "lower bound. It should be a scalar")
.add_argument("clip_max", "Tensor", "upper bound. It should be a scalar")
.set_attrs_type_key("relay.attrs.SimulatedQuantizeAttrs")
.set_support_level(11)
.add_type_rel("SimulatedQuantize", SimulatedQuantizeRel);


TVM_REGISTER_API("relay._quantize.simulated_quantize")
.set_body_typed<Expr(Expr, Expr, Expr, Expr, int, bool, std::string, bool,
                     std::string, std::string, std::string)>(
  [](Expr data, Expr dom_scale, Expr clip_min, Expr clip_max,
     int kind, bool sign, std::string rounding, int passthrough,
     std::string granularity, std::string layout, std::string op_hint) {
    auto attrs = make_node<SimulatedQuantizeAttrs>();
    attrs->kind = kind;
    attrs->sign = sign;
    attrs->rounding = rounding;
    attrs->passthrough = passthrough;
    attrs->granularity = granularity;
    attrs->layout = layout;
    attrs->op_hint = op_hint;
    static const Op& op = Op::Get("relay.op.annotation.simulated_quantize");
    return CallNode::make(op, {data, dom_scale, clip_min, clip_max}, Attrs(attrs), {});
  });


// =============
// annotate pass

Expr QAnnotateExprNode::Realize() const {
  const auto& cfg = QConfig::Current();
  if (cfg->store_lowbit_output) {
    // store low bit output back for VTA
    const PackedFunc* layout_f = runtime::Registry::Get("relay.quantize._get_layout");
    std::string layout = (*layout_f) (this->expr);
    const PackedFunc* f = runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
    return (*f)(this->expr, static_cast<int>(kQInput), layout, (std::string)
this->expr.as<CallNode>()->op.as<OpNode>()->name);
  } else {
    return expr;
  }
}

QAnnotateExpr QAnnotateExprNode::make(Expr expr, QAnnotateKind kind) {
  auto rnode = make_node<QAnnotateExprNode>();
  rnode->expr = expr;
  rnode->kind = kind;
  return QAnnotateExpr(rnode);
}

TVM_REGISTER_API("relay._quantize.make_annotate_expr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = QAnnotateExprNode::make(args[0],
      static_cast<QAnnotateKind>(args[1].operator int()));
  });


/*
TODO(eqy)
TVM_REGISTER_API("relay._quantize.annotate")
.set_body_typed<Expr(Expr)>([] (const Expr& expr) {
  std::function<Expr(const Expr&)> fmulti_ref = [](const Expr& e) {
      if (e->derived_from<TempExprNode>()) {
        const auto* n = e.as<QAnnotateExprNode>();
        CHECK(n);
        const PackedFunc* f = runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
        const PackedFunc* layout_f = runtime::Registry::Get("relay.quantize._get_layout");
        std::string layout = (*layout_f) (n->expr);
        std::string name = n->expr.as<CallNode>()->op.as<OpNode>()->name;
        Expr ret = (*f)(n->expr, static_cast<int>(kQInput), layout, name);
        return static_cast<Expr>(QAnnotateExprNode::make(ret, kQInput));
      }
      return e;
    };
  return ForwardRewrite(expr, "FQAnnotateRewrite", nullptr, fmulti_ref);
});
*/


// =============
// realize pass

Expr InferTypeOpt(const Expr& expr) {
  auto mod = ModuleNode::FromExpr(expr);
  mod = transform::InferType()(mod);
  auto entry_func = mod->Lookup("main");
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

Expr FoldConstantOpt(const Expr& expr) {
  auto mod = ModuleNode::FromExpr(expr);
  mod = transform::FoldConstant()(mod);
  auto entry_func = mod->Lookup("main");
  return expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
}

Expr _ReshapeChannelScale(Expr dom_scale, Expr arr, size_t pos) {
  auto* dom_scale_tensor = dom_scale.as<ConstantNode>();
  CHECK(dom_scale_tensor);
  Array<IndexExpr> data_shape;

  if (!arr->checked_type_.defined()) {
    //arr = InferType(arr, Module(nullptr));
    arr = InferTypeOpt(arr);
    data_shape = arr->checked_type().as<TensorTypeNode>()->shape;
  } else {
    data_shape = arr->checked_type().as<TensorTypeNode>()->shape;
  }
  Array<IndexExpr> dom_scale_shape = dom_scale_tensor->tensor_type()->shape;
  Array<Integer> broadcast_shape;

  CHECK(dom_scale_shape.size() <= 1);
  if (dom_scale_shape[0].as<IntImm>()->value == 1) {
    // leverage implicit broadcasting
    return dom_scale;
  }

  int channels = -1;
  if (dom_scale_shape.size() == 1) {
    channels = dom_scale_shape[0].as<IntImm>()->value;
  }
  for (size_t i = 0; i < data_shape.size(); i++) {
    int dim = data_shape[i].as<IntImm>()->value;
    if (i == pos) {
      CHECK(dim == channels || dim == 1);
      broadcast_shape.push_back(channels);
   } else {
      broadcast_shape.push_back(1);
    }
  }
  return Reshape(dom_scale, broadcast_shape);
}

inline bool _IsTensor(Expr dom_scale) {
  auto* dom_scale_tensor = dom_scale.as<ConstantNode>();
  CHECK(dom_scale_tensor);
  Array<IndexExpr> dom_scale_shape = dom_scale_tensor->tensor_type()->shape;
  if (dom_scale_shape.size() >= 1) {
    CHECK(dom_scale_shape.size() == 1);
    return true;
  }
  return false;
}

int _FindChannelPos(Expr arr, const std::string &layout) {
  Array<IndexExpr> data_shape;
  if (!arr->checked_type_.defined()) {
    //arr = InferType(arr, Module(nullptr));
    arr = InferTypeOpt(arr);
    data_shape = arr->checked_type().as<TensorTypeNode>()->shape;
  } else {
    data_shape = arr->checked_type().as<TensorTypeNode>()->shape;
  }
  // TODO: robust handling of this case
  if (data_shape.size() < layout.size()) {
    return 0;
  }

  int pos = layout.find("C");
  if (pos < 0) {
      pos = layout.find("I");
  }
  return pos;
}

inline bool _ConstantEq(Expr s1, Expr s2) {
  auto* s1_tensor = s1.as<ConstantNode>();
  auto* s2_tensor = s2.as<ConstantNode>();
  CHECK(s1_tensor);
  CHECK(s2_tensor);
  Array<IndexExpr> s1_tensor_shape = s1_tensor->tensor_type()->shape;
  Array<IndexExpr> s2_tensor_shape = s2_tensor->tensor_type()->shape;
  CHECK(s1_tensor_shape.size() == s2_tensor_shape.size());
  // non-vector constants not suported
  CHECK_EQ(s1_tensor_shape.size(), 1);
  if (s1_tensor_shape[0].as<IntImm>()->value != s2_tensor_shape[0].as<IntImm>()->value) {
    size_t dim;
    float val;
    const ConstantNode* tensor;
    if (s1_tensor_shape[0].as<IntImm>()->value == 1) {
      dim = s2_tensor_shape[0].as<IntImm>()->value;
      tensor = s2_tensor;
      val = static_cast<float *>(s1_tensor->data->data)[0];
    } else if (s2_tensor_shape[0].as<IntImm>()->value == 1) {
      dim = s1_tensor_shape[0].as<IntImm>()->value;
      tensor = s2_tensor;
      val = static_cast<float *>(s1_tensor->data->data)[0];
    } else {
      return false;
    }
    for (size_t i = 0; i < dim; i++) {
      if (val !=\
        static_cast<float *>(tensor->data->data)[i])
        return false;
    }
    return true;
  }
  size_t dim = s1_tensor_shape[0].as<IntImm>()->value;
  for (size_t i = 0; i < dim; i++) {
    if (static_cast<float *>(s1_tensor->data->data)[i] !=\
      static_cast<float *>(s2_tensor->data->data)[i])
      return false;
  }
  return true;
}

// Eagerly produce a new ConstantNode by applying an elementwise operation to an
// existing ConstantNode with a custom function
inline Expr _FloatLambda(Expr data, float (*func)(float)) {
    CHECK(_IsTensor(data));
    auto* data_tensor = data.as<ConstantNode>();
    CHECK(data_tensor);
    Array<IndexExpr> data_shape = data_tensor->tensor_type()->shape;
    std::vector<int64_t> new_data_shape;
    CHECK_EQ(data_shape.size(), 1);

    size_t dim = data_shape[0].as<IntImm>()->value;
    new_data_shape.push_back(dim);

    DLContext ctx;
    ctx.device_type = kDLCPU;
    ctx.device_id = 0;

    DLDataType dtype;
    dtype.code = kDLFloat;
    dtype.bits = 32;
    dtype.lanes = 1;

    runtime::NDArray new_data_array = runtime::NDArray::Empty(new_data_shape, dtype, ctx);

    for (size_t i = 0; i < dim; i++) {
        reinterpret_cast<float *>(new_data_array->data)[i] =\
        (*func)(reinterpret_cast<float *>(data_tensor->data->data)[i]);
    }

    return ConstantNode::make(new_data_array);
}

Expr QRealizeIntExprNode::Realize() const {
  const auto& cfg = QConfig::Current();
  Expr data = this->data;
  if (cfg->store_lowbit_output) {
    data = Cast(data, cfg->dtype_input);
  }
  // dequantize
  data = Cast(data, Float(32));
  int pos = _FindChannelPos(data, this->data_layout);
  CHECK(pos >= 0);
  Expr broadcastable_dom_scale = _ReshapeChannelScale(this->dom_scale, data, pos);
  data = Multiply(data, broadcastable_dom_scale);
  return data;
}

QRealizeIntExpr QRealizeIntExprNode::make(Expr data, Expr dom_scale, DataType dtype, std::string data_layout) {
  NodePtr<QRealizeIntExprNode> n = make_node<QRealizeIntExprNode>();
  n->data = std::move(data);
  n->dom_scale = std::move(dom_scale);
  n->dtype = std::move(dtype);
  n->data_layout = std::move(data_layout);
  return QRealizeIntExpr(n);
}


/* calculate `data * s1 / s2`, use shift if possible */
inline Expr MulAndDiv(Expr data, float s1, float s2) {
  // here we assume the dtype of data is dtype activation
  const QConfig& cfg = QConfig::Current();
  if (s1 == s2) return data;

  float factor = s1 / s2;
  float shift_factor = std::log2(factor);
  CHECK_GT(shift_factor, 0);
  if (static_cast<int>(shift_factor) == shift_factor) {
    return LeftShift(data, MakeConstantScalar(cfg->dtype_activation,
                                              static_cast<int>(shift_factor)));
  } else if (static_cast<int>(factor) == factor) {
    return Multiply(data, MakeConstantScalar(cfg->dtype_activation, factor));
  } else {
    LOG(FATAL) << "fall back to float computation";
    data = Cast(data, Float(32));
    return Multiply(data, MakeConstantScalar(Float(32), factor));
  }
}

inline Expr MulAndDiv(Expr data, Expr s1, Expr s2, Expr ref_data, const std::string& layout) {
  // here we assume the dtype of data is dtype activation
  CHECK(_IsTensor(s1));
  CHECK(_IsTensor(s2));
  if (_ConstantEq(s1, s2)) return data;
  // should be constant
  Expr factor = Divide(s1, s2);
  factor = FoldConstantOpt(factor);
  // should be constant
  Expr shift_factor = _FloatLambda(factor, &std::log2f);
  auto* shift_factor_tensor = shift_factor.as<ConstantNode>();
  CHECK(shift_factor_tensor);
  Array<IndexExpr> shift_factor_tensor_shape = shift_factor_tensor->tensor_type()->shape;
  int64_t channels = shift_factor_tensor_shape[0].as<IntImm>()->value;
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  DLDataType dtype;
  dtype.code = kDLInt;
  dtype.bits = 32;
  dtype.lanes = 1;
  runtime::NDArray shift_array = runtime::NDArray::Empty({channels}, dtype, ctx);
  for (int64_t dim = 0; dim < channels; dim++) {
    float cur_shift_factor = static_cast<float*>(shift_factor_tensor->data->data)[dim];
    // currently only support power of two scaling
    CHECK(static_cast<int>(cur_shift_factor) == cur_shift_factor);
    reinterpret_cast<int *>(shift_array->data)[dim] =\
      static_cast<int>(cur_shift_factor);
  }
  int pos = _FindChannelPos(ref_data, layout);
  CHECK(pos >= 0);
  Expr broadcastable_shift = _ReshapeChannelScale(ConstantNode::make(shift_array), ref_data, pos);
  return LeftShift(data, broadcastable_shift);
}

float _RoundBias(float shift_nbit) {
    float round_bias = std::pow(2.0, shift_nbit - 1);
    return round_bias;
}

Expr QuantizeRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  // do not handle data type cast
  const auto param = ref_call->attrs.as<SimulatedQuantizeAttrs>();
  CHECK_EQ(param->rounding, "round");

  Expr dom_scale = new_args[1];
  Expr clip_min = new_args[2];
  Expr clip_max = new_args[3];

  float clip_min_imm = GetScalarFromConstant<float>(clip_min);
  float clip_max_imm = GetScalarFromConstant<float>(clip_max);

  auto* dom_scale_tensor = dom_scale.as<ConstantNode>();
  CHECK(dom_scale_tensor);
  Array<IndexExpr> dom_scale_shape = dom_scale_tensor->tensor_type()->shape;

  std::string layout = param->layout;

  // x * idom_scale = y * odom_scale
  // => y = x * idom_scale / odom_scale
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    // int32->int8
    Expr data = n->data;
    auto* idom_scale_tensor = n->dom_scale.as<ConstantNode>();
    CHECK(idom_scale_tensor);
    Array<IndexExpr> idom_scale_shape = idom_scale_tensor->tensor_type()->shape;
    CHECK(dom_scale_shape.size() == 1); // remove support for floating point scalar case
/* TODO(eqy)
    if (dom_scale_shape.size() >= 1) {
      CHECK(dom_scale_shape.size() == 1);
      CHECK(idom_scale_shape.size() == 1);
      size_t dom_scale_channels = dom_scale_shape[0].as<IntImm>()->value;
      size_t idom_scale_channels = idom_scale_shape[0].as<IntImm>()->value;
      CHECK(dom_scale_channels == idom_scale_channels || dom_scale_channels == 1
            || idom_scale_channels == 1);
      Expr factor = Divide(dom_scale, n->dom_scale);
      factor = FoldConstantOpt(factor);
      auto* factor_tensor = factor.as<ConstantNode>();
      CHECK(factor_tensor != nullptr);
      Expr shift_factor = _FloatLambda(factor, &std::log2f);
      auto* shift_factor_tensor = shift_factor.as<ConstantNode>();
      CHECK(shift_factor_tensor != nullptr);
      size_t dim = shift_factor_tensor->data->shape[0];
      for (size_t i = 0; i < dim; i++) {
        float val = static_cast<float*>(shift_factor_tensor->data->data)[i];
        CHECK(static_cast<int>(val) == val);
      }
      if (cfg->round_for_shift) {
        Expr round_bias = _FloatLambda(shift_factor, _RoundBias);
        round_bias = FoldConstantOpt(round_bias);
        int pos = _FindChannelPos(ref_call->args[0], layout);
        CHECK(pos >= 0);
        round_bias = _ReshapeChannelScale(round_bias, ref_call->args[0], pos);
        round_bias = FoldConstantOpt(round_bias);
        CHECK(round_bias.as<ConstantNode>() != nullptr);
        round_bias = FoldConstantOpt(round_bias);
        round_bias = Cast(round_bias, n->dtype);
        // TODO: why can we not use cfg->dtype_activation?
        data = Add(data, round_bias);
      }
 */   
    CHECK(dom_scale_shape.size() == 1);
    CHECK(idom_scale_shape.size() == 1);
    size_t dom_scale_channels = dom_scale_shape[0].as<IntImm>()->value;
    size_t idom_scale_channels = idom_scale_shape[0].as<IntImm>()->value;
    CHECK(dom_scale_channels == idom_scale_channels || dom_scale_channels == 1
          || idom_scale_channels == 1);
    Expr factor = Divide(dom_scale, n->dom_scale);
    factor = FoldConstantOpt(factor);
    auto* factor_tensor = factor.as<ConstantNode>();
    CHECK(factor_tensor != nullptr);
    Expr shift_factor = _FloatLambda(factor, &std::log2f);
    auto* shift_factor_tensor = shift_factor.as<ConstantNode>();
    CHECK(shift_factor_tensor != nullptr);
    size_t dim = shift_factor_tensor->data->shape[0];
    for (size_t i = 0; i < dim; i++) {
      float val = static_cast<float*>(shift_factor_tensor->data->data)[i];
      CHECK(static_cast<int>(val) == val);
    }
    if (cfg->round_for_shift) {
      Expr round_bias = _FloatLambda(shift_factor, _RoundBias);
      round_bias = FoldConstantOpt(round_bias);
      int pos = _FindChannelPos(ref_call->args[0], layout);
      CHECK(pos >= 0);
      round_bias = _ReshapeChannelScale(round_bias, ref_call->args[0], pos);
      round_bias = FoldConstantOpt(round_bias);
      CHECK(round_bias.as<ConstantNode>() != nullptr);
      round_bias = FoldConstantOpt(round_bias);
      round_bias = Cast(round_bias, n->dtype);
      // TODO(eqy): why can we not use cfg->dtype_activation?
      data = Add(data, round_bias);
    }
    int pos = _FindChannelPos(ref_call->args[0], layout);
    CHECK(pos >= 0);
    shift_factor = _ReshapeChannelScale(shift_factor, data, pos);
    shift_factor = Cast(shift_factor, n->dtype);
    data = RightShift(data, shift_factor);
    data = Clip(data, clip_min_imm, clip_max_imm);
    Expr res = QRealizeIntExprNode::make(data, dom_scale, n->dtype, layout);
    return res;
  }

  // quantize from real
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  Expr data = new_args[0];
  Expr scaled_data;
  CHECK(dom_scale_shape.size() >= 1);
  CHECK(dom_scale_shape.size() == 1);
  int pos = _FindChannelPos(ref_call->args[0], layout);
  CHECK(pos >= 0);
  Expr broadcastable_dom_scale = _ReshapeChannelScale(dom_scale, new_args[0], pos);
  scaled_data = Multiply(data, Divide(MakeConstantScalar(Float(32), 1),
                                           broadcastable_dom_scale));
  Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
  return QRealizeIntExprNode::make(round_data, dom_scale, Float(32), layout);
}



RELAY_REGISTER_OP("relay.op.annotation.simulated_quantize")
.set_attr<FForwardRewrite>("FQRealizeRewrite", QuantizeRealize);

bool _IsStridedSlice(Expr arg) {
  auto ref_arg = arg.as<CallNode>();
  if (ref_arg && ref_arg->op == Op::Get("strided_slice")) {
    return true;
  }
  return false;
}

void _GetStridedIdx(Expr arg, std::vector<size_t> &idx) {
  auto ref_arg = arg.as<CallNode>();
  auto param = ref_arg->attrs.as<StridedSliceAttrs>();
  for (size_t i = 0; i < param->begin.size(); i++) {
    auto intimm1 = param->begin[i].as<IntImm>();
    auto intimm2 = param->end[i].as<IntImm>();
    if (!intimm1 || !intimm2) {
        continue;
    }
    if (intimm2->value - intimm1->value == 0) {
        continue;
    }
    CHECK(intimm1->value >= 0);
    CHECK(intimm1->value >= 0);
    idx.push_back(intimm1->value);
    idx.push_back(intimm2->value);
  }
}

Expr Conv2dRealize(const Call& ref_call,
                   const Array<Expr>& new_args,
                   const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  CHECK(lhs);
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
  CHECK(rhs);

  Expr ldata = lhs->data;
  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = rhs->data;
  rdata = Cast(rdata, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<Conv2DAttrs>();
  auto attrs = make_node<Conv2DAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr input_scale = lhs->dom_scale;
  Expr weight_scale = rhs->dom_scale;

  Array<IndexExpr> data_shape = input_scale.as<ConstantNode>()->tensor_type()->shape;
  Array<IndexExpr> weight_shape = weight_scale.as<ConstantNode>()->tensor_type()->shape;
  CHECK(data_shape.size() == 1);
  CHECK(weight_shape.size() == 1);
  size_t data_dim = data_shape[0].as<IntImm>()->value;
  size_t weight_dim = weight_shape[0].as<IntImm>()->value;

  Expr dom_scale;
  /* Special handling for strided_slice is needed because it changes the number
   * of channel dimensions and the number of per-channel scales. We may consider
   * changing the srided_slice rewrite to something other than identity to avoid
   * this issue.*/
  if (data_dim == weight_dim) {
    // TODO)eqy): special handling for only layer wise scale (when both scales are size 1), we can skip this
    // calculation and only do the old style:
    auto* data_scale_tensor = input_scale.as<ConstantNode>();
    auto* weight_scale_tensor = weight_scale.as<ConstantNode>();

    // CURRENT scheme relies product and weight scales to be matched after
    // multiplying
    float max_output_scale =\
    reinterpret_cast<float*>(data_scale_tensor->data->data)[0]*\
    reinterpret_cast<float*>(weight_scale_tensor->data->data)[0];

    for (size_t i = 0; i < weight_dim; i++) {
      float cur_output_scale =\
      reinterpret_cast<float*>(data_scale_tensor->data->data)[i]*\
      reinterpret_cast<float*>(weight_scale_tensor->data->data)[i];
      CHECK(cur_output_scale == max_output_scale);
    }
    dom_scale = Multiply(Ones({1}, Float(32)), MakeConstantScalar(Float(32), max_output_scale));
    dom_scale = FoldConstantOpt(dom_scale);

    CHECK(dom_scale.as<ConstantNode>());
    CHECK(dom_scale.as<ConstantNode>()->tensor_type()->shape.size() == 1);
  } else {
    // depthwise
    CHECK(weight_dim == 1);

    // unmatched scales are fine for depthwise convolution
    dom_scale = Multiply(input_scale, weight_scale);
    dom_scale = FoldConstantOpt(dom_scale);
    CHECK(dom_scale.as<ConstantNode>());
    CHECK((size_t) dom_scale.as<ConstantNode>()->tensor_type()->shape[0].as<IntImm>()->value == data_dim);
  }
  Expr ret = CallNode::make(ref_call->op,
    {ldata, rdata}, Attrs(attrs), ref_call->type_args);
/*
  TODO(eqy)
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExprNode::make(ret, dom_scale, out_dtype);
*/

  return QRealizeIntExprNode::make(ret, dom_scale, out_dtype, attrs->data_layout);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", Conv2dRealize);

Expr DenseRealize(const Call& ref_call,
                  const Array<Expr>& new_args,
                  const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->derived_from<TempExprNode>() || !new_args[1]->derived_from<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
  CHECK(lhs);
  CHECK(rhs);

  Expr ldata = lhs->data;
  if (lhs->dtype != cfg->dtype_input) {
    ldata = Cast(ldata, cfg->dtype_input);
  }
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);
  const auto ref_attrs = ref_call->attrs.as<DenseAttrs>();
  CHECK(ref_attrs);
  auto attrs = make_node<DenseAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;
  Expr ret = CallNode::make(ref_call->op,
          {ldata, rdata}, Attrs(attrs), ref_call->type_args);
/*
  Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
  Expr dom_scale = FoldConstantOpt(mul);
  return QRealizeIntExprNode::make(ret, dom_scale, out_dtype);
*/
  Expr dom_scale = FoldConstantOpt(Multiply(lhs->dom_scale, rhs->dom_scale));
  CHECK(dom_scale.as<ConstantNode>());
  //CHECK(ref_call->args[0].as<QRealizeIntExprNode>());
  const PackedFunc* layout_f = runtime::Registry::Get("relay.quantize._get_layout");
  std::string layout = (*layout_f) (ref_call);

  return QRealizeIntExprNode::make(ret, dom_scale, out_dtype, layout);
}

RELAY_REGISTER_OP("nn.dense")
.set_attr<FForwardRewrite>("FQRealizeRewrite", DenseRealize);

Expr MulRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    // execute the operation with activation data type.
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;
    Expr rdata = rhs->data;

    DataType dtype = cfg->dtype_activation;
    if (lhs->dtype == Float(32)) {
      ldata = Cast(ldata, dtype);
    } else {
      CHECK_EQ(lhs->dtype, dtype);
    }
    if (rhs->dtype == Float(32)) {
      rdata = Cast(rdata, dtype);
    } else {
      CHECK_EQ(rhs->dtype, dtype);
    }

    Expr ret = ForwardOp(ref_call, {ldata, rdata});
   /*
   TODO(eqy): check
    Expr mul = Multiply(lhs->dom_scale, rhs->dom_scale);
    Expr dom_scale = FoldConstantOpt(mul);
    return QRealizeIntExprNode::make(ret, dom_scale, dtype);
   */
    Expr dom_scale = FoldConstantOpt(Multiply(lhs->dom_scale, rhs->dom_scale));
    CHECK(dom_scale.as<ConstantNode>());
    CHECK(dom_scale.as<ConstantNode>()->tensor_type()->shape.size() == 1);
    const PackedFunc* layout_f = runtime::Registry::Get("relay.quantize._get_layout");
    std::string layout = (*layout_f) (ref_call);

    return QRealizeIntExprNode::make(ret, dom_scale, dtype, layout);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("multiply")
.set_attr<FForwardRewrite>("FQRealizeRewrite", MulRealize);


Expr ChooseDomScale(const std::vector<const QRealizeIntExprNode*>& nptrs,
                    bool max=false) {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;

  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;
  if (nptrs.size() == 2) {
    // x = a * s1, y = b * s2
    // x + y = (a * s1 / s2 + b) * s2, if s1 > s2
    //       = (a + b * s2 / s1) * s1, if s2 > s1
    Expr s1 = nptrs[0]->dom_scale;
    Expr s2 = nptrs[1]->dom_scale;
    auto* s1_tensor = s1.as<ConstantNode>();
    auto* s2_tensor = s2.as<ConstantNode>();
    CHECK(s1_tensor);
    CHECK(s2_tensor);
    Array<IndexExpr> s1_shape = s1_tensor->tensor_type()->shape;
    Array<IndexExpr> s2_shape = s2_tensor->tensor_type()->shape;
    // tensor dom scales
    CHECK(s1_shape.size() >= 1);
    CHECK(s1_shape.size() == 1);
    CHECK(s2_shape.size() == 1);
    // broadcasting
    if (s1_shape[0].as<IntImm>()->value != s2_shape[0].as<IntImm>()->value) {
      CHECK(s1_shape[0].as<IntImm>()->value == 1 || s2_shape[0].as<IntImm>()->value == 1);
      const ConstantNode* single;
      const ConstantNode* broadcast_to;
      if (s1_shape[0].as<IntImm>()->value == 1) {
          single = s1_tensor;
          broadcast_to = s2_tensor;
      }
      else {
          single = s2_tensor;
          broadcast_to = s1_tensor;
      }
      float cur_s1 = reinterpret_cast<float*>(single->data->data)[0];
      int64_t dim = broadcast_to->tensor_type()->shape[0].as<IntImm>()->value;

      runtime::NDArray s = runtime::NDArray::Empty({dim}, dtype, ctx);
      for (int64_t i = 0; i < dim; i++) {
        float cur_s2 = reinterpret_cast<float*>(broadcast_to->data->data)[i];
        float cur_s = cur_s1 > cur_s2 ? cur_s2 : cur_s1;
        reinterpret_cast<float*>(s->data)[i] = cur_s;
      }
      return ConstantNode::make(s);
    } else {
      int64_t dim = s1_shape[0].as<IntImm>()->value;
      runtime::NDArray s = runtime::NDArray::Empty({dim}, dtype, ctx);
      for (int64_t i = 0; i < dim; i++) {
        float cur_s1 = reinterpret_cast<float*>(s1_tensor->data->data)[i];
        float cur_s2 = reinterpret_cast<float*>(s2_tensor->data->data)[i];
        reinterpret_cast<float*>(s->data)[i] = cur_s1 > cur_s2 ? cur_s2 : cur_s1;
      }
      return ConstantNode::make(s);
    }
  } else if (max) {
    Expr scale;
    std::vector<float> scales;
    for (size_t i = 0; i < nptrs.size(); i++) {
      Expr s = nptrs[i]->dom_scale;
      auto* s_tensor = s.as<ConstantNode>();
      CHECK(s_tensor);
      Array<IndexExpr> s_shape = s_tensor->tensor_type()->shape;
      CHECK_EQ(s_shape[0].as<IntImm>()->value, 1);
      scales.push_back(static_cast<float*>(s_tensor->data->data)[0]);
      if (!i) {
        scale = s;
      } else {
        scale = Min(s, scale);
      }
    }
    return FoldConstantOpt(scale);
  } else {
    LOG(INFO) << "WARNING, using global scale";
    const QConfig& cfg = QConfig::Current();
    float scale = cfg->global_scale;
    scale = scale / std::pow(2.0, cfg->nbit_activation - 1);
    runtime::NDArray s = runtime::NDArray::Empty({1}, dtype, ctx);
    reinterpret_cast<float*>(s->data)[0] = scale;
    Expr scale_constant = ConstantNode::make(s);
    return scale_constant;
  }
}

/* \brief Unify the dom scale of arguments */
Array<Expr> UnifyDTypeScale(const Array<Expr>& ref_args,
                            const Array<Expr>& args,
                            DataType* dtype_ptr,
                            Expr* scale_ptr,
                            const std::string& layout,
                            bool min=false) {
  static const Op& simulated_quantize = Op::Get("relay.op.annotation.simulated_quantize");
  const QConfig& cfg = QConfig::Current();

  std::vector<const QRealizeIntExprNode*> nptrs;
  Array<Expr> ret;
  for (auto arg : args) {
    const auto* nptr = arg.as<QRealizeIntExprNode>();
    CHECK(nptr);
    nptrs.push_back(nptr);
    ret.push_back(nptr->data);
  }

  // unify the data type
  CHECK_EQ(ref_args.size(), args.size());
  DataType dtype;
  if (ret.size() == 2 && nptrs[1]->dtype == cfg->dtype_input) {
    dtype = cfg->dtype_input;
  } else {
    dtype = cfg->dtype_activation;
  }
  for (size_t i = 0; i < ret.size(); ++i) {
    auto ref_arg = ref_args[i].as<CallNode>();
    if (nptrs[i]->dtype != dtype) {
      ret.Set(i, Cast(ret[i], dtype));
    } else if (ref_arg && ref_arg->op.same_as(simulated_quantize) &&
               ref_arg->attrs.as<SimulatedQuantizeAttrs>()->kind == kQInput) {
      auto new_arg = Cast(ret[i], cfg->dtype_input);
      new_arg = StopFusion(new_arg);
      ret.Set(i, Cast(new_arg, dtype));
    }
  }

  // unify the dom_scale
  // s should be a constant, created by ChooseDomScale
  Expr dom_scale = ChooseDomScale(nptrs, min);
  for (size_t i = 0; i < ret.size(); ++i) {
    Expr cur_s = nptrs[i]->dom_scale;
    ret.Set(i, MulAndDiv(ret[i], cur_s, dom_scale, ref_args[i], layout));
  }

  *dtype_ptr = dtype;
  *scale_ptr = dom_scale;
  return ret;
}


Expr AddRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    const PackedFunc* layout_f = runtime::Registry::Get("relay.quantize._get_layout");
    std::string layout = (*layout_f) (ref_call);
    Array<Expr> ret_args = UnifyDTypeScale(ref_call->args, new_args, &dtype, &dom_scale, layout);
    Expr ret = ForwardOp(ref_call, ret_args);
    return QRealizeIntExprNode::make(ret, dom_scale, dtype, layout);
  }

  CHECK(!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>());
  return Expr(nullptr);
}


RELAY_REGISTER_OP("add")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AddRealize);

Expr ClipRealize(const Call& ref_call,
                 const Array<Expr>& new_args,
                 const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    const auto ref_attrs = ref_call->attrs.as<ClipAttrs>();
    auto attrs = make_node<ClipAttrs>();
    double dom_scale = GetScalarFromConstant<float>(n->dom_scale);
    attrs->a_min = ref_attrs->a_min / dom_scale;
    attrs->a_max = ref_attrs->a_max / dom_scale;

    Expr ret = CallNode::make(ref_call->op,
      {n->data}, Attrs(attrs), ref_call->type_args);
    return QRealizeIntExprNode::make(ret, n->dom_scale, n->dtype, n->data_layout);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("clip")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ClipRealize);


/* \brief Unify the dom scale of arguments */
Array<Expr> ConcatenateDTypeScale(const Array<Expr>& ref_args,
                                  const Array<Expr>& args,
                                  DataType* dtype_ptr,
                                  Expr* scale_ptr,
                                  const std::string& layout) {
  static const Op& simulated_quantize = Op::Get("relay.op.annotation.simulated_quantize");
  const QConfig& cfg = QConfig::Current();

  std::vector<const QRealizeIntExprNode*> nptrs;
  Array<Expr> ret;
  for (auto arg : args) {
    const auto* nptr = arg.as<QRealizeIntExprNode>();
    CHECK(nptr);
    nptrs.push_back(nptr);
    ret.push_back(nptr->data);
  }
  // unify the data type
  CHECK_EQ(ref_args.size(), args.size());
  DataType dtype = cfg->dtype_activation;
  for (size_t i = 0; i < ret.size(); ++i) {
    auto ref_arg = ref_args[i].as<CallNode>();
    if (nptrs[i]->dtype != dtype) {
      ret.Set(i, Cast(ret[i], dtype));
    } else if (ref_arg && ref_arg->op.same_as(simulated_quantize) &&
               ref_arg->attrs.as<SimulatedQuantizeAttrs>()->kind == kQInput) {
      auto new_arg = Cast(ret[i], cfg->dtype_input);
      //TODO(eqy): if (cfg->use_stop_fusion) {
      //  new_arg = StopFusion(new_arg);
      //}
      ret.Set(i, Cast(new_arg, dtype));
    }
  }
  // unify the dom_scale
  // s should be a constant, created by ChooseDomScale
  Array<Expr> dom_scales;
  for (size_t i = 0; i < ret.size(); ++i) {
    Expr data = ref_args[i];
    if (!data->checked_type_.defined()) {
      //data = InferType(data, Module(nullptr));
      data = InferTypeOpt(data);
    }
    int pos = _FindChannelPos(data, layout);
    int dom_scale_dim = nptrs[i]->dom_scale.as<ConstantNode>()->tensor_type()->shape[0].as<IntImm>()->value;
    int channels = data->checked_type().as<TensorTypeNode>()->shape[pos].as<IntImm>()->value;
    if (channels != dom_scale_dim) {
      CHECK(dom_scale_dim == 1);
      dom_scales.push_back(FoldConstantOpt(Multiply(Ones({channels}, Float(32)), nptrs[i]->dom_scale)));
    } else {
      dom_scales.push_back(nptrs[i]->dom_scale);
    }
  }
  Expr dom_scale = MakeConcatenate(TupleNode::make(dom_scales), 0);
  dom_scale = FoldConstantOpt(dom_scale);
  *dtype_ptr = dtype;
  *scale_ptr = dom_scale;
  return ret;
}


Expr ConcatenateRealize(const Call& ref_call,
                        const Array<Expr>& new_args,
                        const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  CHECK_EQ(ref_call->args.size(), 1);
  const auto* tuple = new_args[0].as<TupleNode>();
  const auto* ref_tuple = ref_call->args[0].as<TupleNode>();
  CHECK(ref_tuple);
  CHECK(tuple);
  const Array<Expr>& arr = tuple->fields;
  const Array<Expr>& ref_arr = ref_tuple->fields;

  if (arr[0].as<QRealizeIntExprNode>()) {
    DataType dtype;
    Expr dom_scale;
    // CHECK that it is is a per-channel concatenate
    // TODO(eqy): consider adding granularity as a field instead of relying on
    // brittle heuristic
    if (arr[0].as<QRealizeIntExprNode>()->dom_scale.as<ConstantNode>()->tensor_type()->shape[0].as<IntImm>()->value > 1) {
      Array<Expr> ret_args = ConcatenateDTypeScale(ref_arr, arr, &dtype, &dom_scale, arr[0].as<QRealizeIntExprNode>()->data_layout);
      Expr ret = ForwardOp(ref_call, {TupleNode::make(ret_args)});
      return QRealizeIntExprNode::make(ret, dom_scale, dtype, arr[0].as<QRealizeIntExprNode>()->data_layout);

    } else {
    Array<Expr> ret_args = UnifyDTypeScale(ref_arr, arr, &dtype, &dom_scale, arr[0].as<QRealizeIntExprNode>()->data_layout, true);
    Expr ret = ForwardOp(ref_call, {TupleNode::make(ret_args)});
    return QRealizeIntExprNode::make(ret, dom_scale, dtype, arr[0].as<QRealizeIntExprNode>()->data_layout);
    }
  } else {
    for (auto arg : new_args) {
      CHECK(!arg->derived_from<TempExprNode>());
    }
    return Expr(nullptr);
  }
}


RELAY_REGISTER_OP("concatenate")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ConcatenateRealize);


/* \brief forward the original operator */
Expr IdentityRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    int scale_dim = n->dom_scale.as<ConstantNode>()->tensor_type()->shape[0].as<IntImm>()->value;
    // TODO(eqy):use more reliable check for per-layer scale
    if (ref_call->op == Op::Get("strided_slice") && scale_dim > 1) {
      std::vector<size_t> idx;
      _GetStridedIdx(ref_call, idx);
      Expr sliced_scale = MakeStridedSlice(n->dom_scale, {(int) idx[0]}, {(int) idx[1]}, {1});
      sliced_scale = FoldConstantOpt(sliced_scale);
      Expr ret = ForwardOp(ref_call, {n->data});
      return QRealizeIntExprNode::make(ret, sliced_scale, n->dtype, n->data_layout);
    } else {
      Expr ret = ForwardOp(ref_call, {n->data});
      return QRealizeIntExprNode::make(ret, n->dom_scale, n->dtype, n->data_layout);
    }
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}


RELAY_REGISTER_OP("nn.relu")
.set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("strided_slice")
.set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

RELAY_REGISTER_OP("annotation.stop_fusion")
.set_attr<FForwardRewrite>("FQRealizeRewrite", IdentityRealize);

/* \brief for unary operators which requantize its input to dtype_nbit */
Expr CastDtypeInputRealize(const Call& ref_call,
                           const Array<Expr>& new_args,
                           const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = Cast(n->data, cfg->dtype_input);
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExprNode::make(ret, n->dom_scale, cfg->dtype_input, n->data_layout);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.max_pool2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", CastDtypeInputRealize);


Expr AvgPoolRealize(const Call& ref_call,
                    const Array<Expr>& new_args,
                    const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = n->data;
    if (n->dtype != cfg->dtype_activation) {
      data = Cast(n->data, cfg->dtype_activation);
    }
    Expr ret = ForwardOp(ref_call, {data});
    return QRealizeIntExprNode::make(ret, n->dom_scale, cfg->dtype_activation, n->data_layout);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.avg_pool2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AvgPoolRealize);

Expr ForceCastRealize(const Call& ref_call,
                      const Array<Expr>& new_args,
                      const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = Cast(n->data, cfg->dtype_input);
    return QRealizeIntExprNode::make(ret, n->dom_scale, cfg->dtype_input, n->data_layout);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("annotation.force_cast")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ForceCastRealize);

TVM_REGISTER_API("relay._quantize.realize")
.set_body_typed<Expr(Expr)>([](const Expr& e) {
  Expr ret = ForwardRewrite(e, "FQRealizeRewrite", nullptr, nullptr);
  return ret;
});


// =============
// qconfig

QConfig qconfig() {
  return QConfig(make_node<QConfigNode>());
}

/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMQConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  QConfig default_config;

  /*! \brief The current build config context */
  std::stack<QConfig> context_stack;

  TVMQConfigThreadLocalEntry() :
    default_config(qconfig()) {
  }
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMQConfigThreadLocalEntry> TVMQConfigThreadLocalStore;

void QConfig::EnterQConfigScope(const QConfig& build_config) {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.push(build_config);
}

void QConfig::ExitQConfigScope() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.pop();
}

QConfig& QConfig::Current() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }

  return entry->default_config;
}

TVM_REGISTER_NODE_TYPE(QConfigNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<QConfigNode>([](const QConfigNode *op, IRPrinter *p) {
  p->stream << "qconfig(";
  p->stream << "nbit_input=" << op->nbit_input << ", ";
  p->stream << "nbit_weight=" << op->nbit_weight << ", ";
  p->stream << "nbit_activation=" << op->nbit_activation << ", ";
  p->stream << "global_scale=" << op->global_scale << ", ";
  p->stream << "skip_conv_layers==" << op->skip_conv_layers << ", ";
  p->stream << "round_for_shift==" << op->round_for_shift << ", ";
  p->stream << "store_lowbit_output==" << op->store_lowbit_output << ", ";
  p->stream << "debug_enabled_ops==" << op->debug_enabled_ops;
  p->stream << "passthrough_bound=" << op->passthrough_bound << ", ";
  //TODO(eqy): p->stream << "use_stop_fusion==" << op->use_stop_fusion << ", ";
  p->stream << "granularity="<< op->granularity;
  p->stream << ")";
  p->stream << ")";
});

TVM_REGISTER_API("relay._quantize._GetCurrentQConfig")
.set_body_typed(QConfig::Current);

TVM_REGISTER_API("relay._quantize._EnterQConfigScope")
.set_body_typed(QConfig::EnterQConfigScope);

TVM_REGISTER_API("relay._quantize._ExitQConfigScope")
.set_body_typed(QConfig::ExitQConfigScope);

Pass QuantizeAnnotate() {
  std::function<Expr(const Expr&)> fmulti_ref = [](const Expr& e) {
    if (e->derived_from<TempExprNode>()) {
      const auto* n = e.as<QAnnotateExprNode>();
      CHECK(n);
      const PackedFunc* f =
          runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
      Expr ret = (*f)(n->expr, static_cast<int>(kQInput));
      return static_cast<Expr>(QAnnotateExprNode::make(ret, kQInput));
    }
    return e;
  };

  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      auto func = Downcast<Function>(ForwardRewrite(f, "FQAnnotateRewrite", nullptr, fmulti_ref));
      auto new_params = func->params;
      for (const auto& x : FreeVars(func)) {
        new_params.push_back(x);
      }
      return FunctionNode::make(new_params,
                                func->body,
                                func->ret_type,
                                func->type_params,
                                func->attrs);
  };
  return CreateFunctionPass(pass_func, 1, "QuantizeAnnotate", {});
}

TVM_REGISTER_API("relay._quantize.QuantizeAnnotate")
.set_body_typed(QuantizeAnnotate);

Pass QuantizeRealizePass() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(
          ForwardRewrite(f, "FQRealizeRewrite", nullptr, nullptr));
  };
  return CreateFunctionPass(pass_func, 1, "QuantizeRealize", {});
}

TVM_REGISTER_API("relay._quantize.QuantizeRealize")
.set_body_typed(QuantizeRealizePass);

Pass QuantizeRewriteForVTAPass() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(
          ForwardRewrite(f, "FQVTARewrite", nullptr, nullptr));
  };
  return CreateFunctionPass(pass_func, 1, "QuantizeRewriteForVTA", {});
}

TVM_REGISTER_API("relay._quantize.QuantizeRewriteForVTA")
.set_body_typed(QuantizeRewriteForVTAPass);

// =============
// Insert stop_fusion for vta.


Expr QVTAExprNode::Realize() const {
  Expr ret = ForceCast(this->expr);
  return StopFusion(ret);
}

QVTAExpr QVTAExprNode::make(Expr expr) {
  auto rnode = make_node<QVTAExprNode>();
  rnode->expr = expr;
  return QVTAExpr(rnode);
}

TVM_REGISTER_API("relay._quantize.make_vta_expr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = QVTAExprNode::make(args[0]);
  });

TVM_REGISTER_API("relay._quantize.make_stop_fusion")
.set_body_typed<Expr(Expr)>([] (const Expr& expr) {
  return StopFusion(expr);
});

TVM_REGISTER_API("relay._quantize.temp_expr_realize")
.set_body_typed<Expr(Expr)>([] (const Expr& expr) {
  const QVTAExprNode* n = expr.as<QVTAExprNode>();
  CHECK(n);
  return n->Realize();
});


}  // namespace quantize
}  // namespace relay
}  // namespace tvm
