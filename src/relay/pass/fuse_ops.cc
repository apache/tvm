/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file src/tvm/relay/pass/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#include <tvm/ir_operator.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

// Simple fuser that only makes each operator function as primitive.
class SimpleFuser : public ExprMutator {
 public:
  // Skip primitive function.
  Expr VisitExpr_(const FunctionNode* fn_node) {
    NodeRef res = FunctionGetAttr(GetRef<Function>(fn_node), "Primitive");
    const ir::IntImm* pval = res.as<ir::IntImm>();
    if (pval && pval->value != 0) {
      return GetRef<Expr>(fn_node);
    } else {
      return ExprMutator::VisitExpr_(fn_node);
    }
  }

  Expr VisitExpr_(const CallNode* call) {
    if (call->op.as<OpNode>()) {
      // Placeholder fusion algorithm which abstracts
      // single definitions into functions only.
      Array<Var> params;
      Array<Expr> inner_args;
      Array<Expr> args;

      int param_number = 0;
      for (auto arg : call->args) {
        std::ostringstream os;
        os << "p" << param_number++;
        auto type = arg->checked_type();
        auto var = VarNode::make(os.str(), type);
        params.push_back(var);
        inner_args.push_back(var);
        args.push_back(this->Mutate(arg));
      }
      auto body = CallNode::make(call->op, inner_args, call->attrs);
      auto func = FunctionNode::make(
          params, body, call->checked_type(), {});
      func = FunctionSetAttr(func, "Primitive", tvm::Integer(1));
      return CallNode::make(func, args, Attrs());
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }
};


Expr FuseOps(const Expr& expr) {
  // First we convert all chains of fusable ops into
  // abstracted functions which we mark as primtive
  // then we convert these primtive functions into
  // new operators.
  return SimpleFuser().Mutate(expr);
}

TVM_REGISTER_API("relay._ir_pass.FuseOps")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = FuseOps(args[0]);
});
}  // namespace relay
}  // namespace tvm
