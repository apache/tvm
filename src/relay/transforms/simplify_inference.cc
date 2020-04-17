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
 * \file simplify_inference.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

Expr BatchNormToInferUnpack(const Attrs attrs,
                            Expr data,
                            Expr gamma,
                            Expr beta,
                            Expr moving_mean,
                            Expr moving_var,
                            Type tdata) {
  auto ttype = tdata.as<TensorTypeNode>();
  CHECK(ttype);
  const auto param = attrs.as<BatchNormAttrs>();
  Expr epsilon = MakeConstantScalar(ttype->dtype, static_cast<float>(param->epsilon));
  Expr var_add_eps = Add(moving_var, epsilon);
  Expr sqrt_var = Sqrt(var_add_eps);
  Expr scale = Divide(MakeConstantScalar(ttype->dtype, 1.0f), sqrt_var);

  if (param->scale) {
    scale = Multiply(scale, gamma);
  }
  Expr neg_mean = Negative(moving_mean);
  Expr shift = Multiply(neg_mean, scale);
  if (param->center) {
    shift = Add(shift, beta);
  }

  auto ndim = ttype->shape.size();
  int axis = (param->axis < 0) ? param->axis + ndim : param->axis;
  scale = ExpandBiasToMatchAxis(scale, ndim, {axis});
  shift = ExpandBiasToMatchAxis(shift, ndim, {axis});

  Expr out = Multiply(data, scale);
  out = Add(out, shift);
  return out;
}


Expr GroupNormToInferUnpack(const Attrs attrs,
                            Expr data,
                            Expr gamma,
                            Expr beta,
                            Type tdata) {
  auto ttype = tdata.as<TensorTypeNode>();
  CHECK(ttype);
  const auto param = attrs.as<GroupNormAttrs>();
  CHECK(param);

  int ndim = ttype->shape.size();
  int axis = (param->axis < 0) ? param->axis + ndim : param->axis;
  Array<Integer> reduced_axes;
  Array<Integer> new_shape;
  Array<Integer> old_shape;

  int num_groups = param->num_groups;
  int channel = ttype->shape[axis].as<IntImmNode>()->value;

  // old_shape = N, C, H, W
  // new shape = N, num_groups, C/num_groups, H, W
  // reduce_axes = axis of (C/num_groups, H, W)
  for (int i = 0; i < ndim; ++i) {
      auto val = ttype->shape[i].as<IntImmNode>()->value;

      // Save the old shape to reshape later
      old_shape.push_back(val);
      if (i == axis) {
          new_shape.push_back(num_groups);
          new_shape.push_back(channel / num_groups);
          reduced_axes.push_back(i + 1);
          continue;
      }
      if (i >= axis) {
          reduced_axes.push_back(i + 1);
      }
      new_shape.push_back(val);
  }

  data = Reshape(data, new_shape);

  Expr epsilon = MakeConstantScalar(ttype->dtype, static_cast<float>(param->epsilon));
  Expr mean = Mean(data, {reduced_axes}, true, false);
  Expr var = Variance(data, mean, {reduced_axes}, true, false);
  Expr denom = Sqrt(Add(var, epsilon));
  Expr out = Divide(Subtract(data, mean), denom);

  out = Reshape(out, old_shape);

  if (param->scale) {
    out = Multiply(out, ExpandBiasToMatchAxis(gamma, ndim, {axis}));
  }
  if (param->center) {
    out = Add(out, ExpandBiasToMatchAxis(beta, ndim, {axis}));
  }

  return out;
}

Expr LayerNormToInferUnpack(const Attrs attrs,
                            Expr data,
                            Expr gamma,
                            Expr beta,
                            Type tdata) {
  auto ttype = tdata.as<TensorTypeNode>();
  CHECK(ttype);
  const auto param = attrs.as<LayerNormAttrs>();
  CHECK(param);

  Expr epsilon = MakeConstantScalar(ttype->dtype, static_cast<float>(param->epsilon));
  Expr mean = Mean(data, {param->axis}, true, false);
  Expr var = Variance(data, mean, {param->axis}, true, false);
  Expr denom = Sqrt(Add(var, epsilon));
  Expr out = Divide(Subtract(data, mean), denom);

  size_t ndim = ttype->shape.size();
  int axis = (param->axis < 0) ? param->axis + ndim : param->axis;
  if (param->scale) {
    out = Multiply(out, ExpandBiasToMatchAxis(gamma, ndim, {axis}));
  }
  if (param->center) {
    out = Add(out, ExpandBiasToMatchAxis(beta, ndim, {axis}));
  }
  return out;
}

Expr InstanceNormToInferUnpack(const Attrs attrs,
                               Expr data,
                               Expr gamma,
                               Expr beta,
                               Type tdata) {
  auto ttype = tdata.as<TensorTypeNode>();
  CHECK(ttype);
  const auto param = attrs.as<InstanceNormAttrs>();
  CHECK(param);

  int ndim = ttype->shape.size();
  int axis = (param->axis < 0) ? param->axis + ndim : param->axis;
  Array<Integer> reduced_axes;
  for (int i = 1; i < ndim; ++i) {
      if (i != axis)
          reduced_axes.push_back(i);
  }

  Expr epsilon = MakeConstantScalar(DataType::Float(32), static_cast<float>(param->epsilon));
  Expr mean = Mean(data, reduced_axes, true, false);
  Expr var = Variance(data, mean, reduced_axes, true, false);
  Expr denom = Sqrt(Add(var, epsilon));
  Expr out = Divide(Subtract(data, mean), denom);

  if (param->scale) {
    out = Multiply(out, ExpandBiasToMatchAxis(gamma, ndim, {axis}));
  }
  if (param->center) {
    out = Add(out, ExpandBiasToMatchAxis(beta, ndim, {axis}));
  }
  return out;
}

Expr L2NormToInferUnpack(const Attrs attrs, Expr data) {
  const auto param = attrs.as<L2NormalizeAttrs>();
  CHECK(param);

  Expr epsilon = MakeConstantScalar(DataType::Float(32), static_cast<float>(param->eps));

  Expr sqr = Multiply(data, data);
  Expr sum = Maximum(Sum(sqr, param->axis, true, false), epsilon);
  Expr sqrt = Sqrt(sum);
  return Divide(data, sqrt);
}

class InferenceSimplifier : public ExprMutator {
 public:
  InferenceSimplifier()
      : batch_norm_op_(Op::Get("nn.batch_norm")),
        dropout_op_(Op::Get("nn.dropout")),
        instance_norm_op_(Op::Get("nn.instance_norm")),
        layer_norm_op_(Op::Get("nn.layer_norm")),
        group_norm_op_(Op::Get("nn.group_norm")),
        l2_norm_op_(Op::Get("nn.l2_normalize")) {}

  Expr VisitExpr_(const TupleGetItemNode* n) final {
    Expr new_e = ExprMutator::VisitExpr_(n);
    const auto* new_n = new_e.as<TupleGetItemNode>();
    if (new_n->index != 0) {
      return new_e;
    }
    if (const auto* call = new_n->tuple.as<CallNode>()) {
      if (call->op == batch_norm_op_) {
        return BatchNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
                                      call->args[3], call->args[4], ty_map_.at(call->args[0]));
      } else if (call->op == dropout_op_) {
        return call->args[0];
      }
    }
    return new_e;
  }

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op == batch_norm_op_) {
      ty_map_[new_n.as<CallNode>()->args[0]] = n->args[0]->checked_type();
    } else if (n->op == layer_norm_op_) {
      const auto* call = new_n.as<CallNode>();
      return LayerNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
                                    n->args[0]->checked_type());
    } else if (n->op == group_norm_op_) {
      const auto* call = new_n.as<CallNode>();
      return GroupNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
                                    n->args[0]->checked_type());
    } else if (n->op == instance_norm_op_) {
      const auto* call = new_n.as<CallNode>();
      return InstanceNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
                                       n->args[0]->checked_type());
    } else if (n->op == l2_norm_op_) {
      const auto* call = new_n.as<CallNode>();
      return L2NormToInferUnpack(call->attrs, call->args[0]);
    }
    return new_n;
  }

 private:
  // Cache the following ops. They will be used in the passes repeatedly for
  // operator equivalence checking so that the registry lookup overhead can be
  // reduced.
  const Op& batch_norm_op_;
  const Op& dropout_op_;
  const Op& instance_norm_op_;
  const Op& layer_norm_op_;
  const Op& group_norm_op_;
  const Op& l2_norm_op_;
  std::unordered_map<Expr, Type, ObjectHash, ObjectEqual> ty_map_;
};

Expr SimplifyInference(const Expr& e) {
  return InferenceSimplifier().Mutate(e);
}

namespace transform {

Pass SimplifyInference() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(SimplifyInference(f));
  };
  return CreateFunctionPass(pass_func, 0, "SimplifyInference", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyInference")
.set_body_typed(SimplifyInference);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
