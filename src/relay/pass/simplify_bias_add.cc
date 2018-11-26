/*!
 * Copyright (c) 2018 by Contributors
 * \file expand_bias_add.cc
 * \brief Expand bias_add to expand_dims and broadcast_add.
 *        This can simplify the passes related to layout (e.g. alter_op_layout).
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class BiasAddSimplifier : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* n) {
    static const Op& bias_add = Op::Get("nn.bias_add");
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op.same_as(bias_add)) {
      Call call = Downcast<Call>(new_n);
      CHECK_EQ(call->args.size(), 2);
      const BiasAddAttrs* param = call->attrs.as<BiasAddAttrs>();

      auto ttype = call->args[0]->type_as<TensorTypeNode>();
      size_t n_dim = ttype->shape.size();
      Expr expanded_bias = ExpandBiasToMatchAxis(call->args[1], n_dim, {param->axis});
      Expr ret = Add(call->args[0], expanded_bias);
      ret->checked_type_ = n->checked_type_;
      return ret;
    }
    return new_n;
  }
};

Expr SimplifyBiasAdd(const Expr& e) {
  return BiasAddSimplifier().Mutate(e);
}

TVM_REGISTER_API("relay._ir_pass.simplify_bias_add")
.set_body([](TVMArgs args, TVMRetValue* ret) {
*ret = SimplifyBiasAdd(args[0]);
});

}  // namespace relay
}  // namespace tvm
