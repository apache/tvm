/*!
 * Copyright (c) 2018 by Contributors
 * \file simplify_inference.cc
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include "./pattern_util.h"

namespace tvm {
namespace relay {

Expr BatchNormToInferUnpack(const Attrs attrs,
                            Expr data,
                            Expr gamma,
                            Expr beta,
                            Expr moving_mean,
                            Expr moving_var,
                            Type tdata) {
  const auto param = attrs.as<BatchNormAttrs>();
  Expr epsilon = MakeConstantScalar(Float(32), static_cast<float>(param->epsilon));
  Expr var_add_eps = Add(moving_var, epsilon);
  Expr sqrt_var = Sqrt(var_add_eps);
  Expr scale = Divide(MakeConstantScalar(Float(32), 1.0f), sqrt_var);

  if (param->scale) {
    scale = Multiply(scale, gamma);
  }
  Expr neg_mean = Negative(moving_mean);
  Expr shift = Multiply(neg_mean, scale);
  if (param->center) {
    shift = Add(shift, beta);
  }

  int axis = param->axis;
  auto ttype = tdata.as<TensorTypeNode>();
  CHECK(ttype);
  auto ndim = ttype->shape.size();
  scale = ExpandBiasToMatchAxis(scale, ndim, {axis});
  shift = ExpandBiasToMatchAxis(shift, ndim, {axis});

  Expr out = Multiply(data, scale);
  out = Add(out, shift);
  return out;
}

class InferenceSimplifier : public ExprMutator {
 public:
  Expr VisitExpr_(const TupleGetItemNode* n) final {
    static const Op& batch_norm = Op::Get("nn.batch_norm");
    static const Op& dropout = Op::Get("nn.dropout");

    Expr new_e = ExprMutator::VisitExpr_(n);
    const auto* new_n = new_e.as<TupleGetItemNode>();
    if (new_n->index != 0) {
      return new_e;
    }
    if (const auto* call = new_n->tuple.as<CallNode>()) {
      if (call->op.same_as(batch_norm)) {
        return BatchNormToInferUnpack(call->attrs, call->args[0], call->args[1], call->args[2],
                                      call->args[3], call->args[4], ty_map_.at(call->args[0]));
      } else if (call->op.same_as(dropout)) {
        return call->args[0];
      }
    }
    return new_e;
  }

  Expr VisitExpr_(const CallNode* n) {
    static const Op& batch_norm = Op::Get("nn.batch_norm");
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op.same_as(batch_norm)) {
      ty_map_[new_n.as<CallNode>()->args[0]] = n->args[0]->checked_type();
    }
    return new_n;
  }

 private:
  std::unordered_map<Expr, Type, NodeHash, NodeEqual> ty_map_;
};

Expr SimplifyInference(const Expr& e) {
  return InferenceSimplifier().Mutate(e);
}

TVM_REGISTER_API("relay._ir_pass.simplify_inference")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = SimplifyInference(args[0]);
  });

}  // namespace relay
}  // namespace tvm
