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
                            Expr moving_var) {
  const auto param = attrs.as<BatchNormAttrs>();
  Expr epsilon = MakeConstantScalar(Float(32), static_cast<float>(param->epsilon));
  Expr var_add_eps = CallNode::make(Op::Get("add"), {moving_var, epsilon});
  Expr sqrt_var = CallNode::make(Op::Get("sqrt"), {var_add_eps});
  Expr scale = CallNode::make(Op::Get("divide"), {MakeConstantScalar(Float(32), 1.0f), sqrt_var});

  if (param->scale) {
    scale = CallNode::make(
      Op::Get("multiply"), {scale, gamma});
  }
  Expr neg_mean = CallNode::make(Op::Get("negative"), {moving_mean});
  Expr shift = CallNode::make(Op::Get("multiply"), {neg_mean, scale});
  if (param->center) {
    shift = CallNode::make(Op::Get("add"), {shift, beta});
  }

  int axis = param->axis;
  const auto* tdata = data->type_as<TensorTypeNode>();
  CHECK(tdata) << "require checked type";
  scale = ExpandBiasToMatchAxis(scale, tdata->shape.size(), {axis});
  shift = ExpandBiasToMatchAxis(shift, tdata->shape.size(), {axis});

  Expr out = CallNode::make(Op::Get("multiply"), {data, scale});
  out = CallNode::make(Op::Get("add"), {out, shift});
  return out;
}

class Simplifier : public ExprMutator {
 public:
  Expr VisitExpr_(const TupleGetItemNode* n) final {
    static const Op& batch_norm = Op::Get("nn.batch_norm");
    static const Op& dropout = Op::Get("nn.dropout");

    Expr new_e = ExprMutator::VisitExpr_(n);
    const auto* new_n = new_e.as<TupleGetItemNode>();
    if (const auto* call = new_n->tuple.as<CallNode>()) {
      if (call->op.same_as(batch_norm)) {
        return BatchNormToInferUnpack(call->attrs,
          call->args[0], call->args[1], call->args[2], call->args[3], call->args[4]);
      } else if (call->op.same_as(dropout)) {
        return call->args[0];
      }
    }
    return new_e;
  }
};

Expr SimplifyInference(const Expr& e) {
  return Simplifier().Mutate(e);
}

TVM_REGISTER_API("relay._ir_pass.simplify_inference")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = SimplifyInference(args[0]);
  });

}  // namespace relay
}  // namespace tvm
