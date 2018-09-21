/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file src/tvm/relay/pass/fuse_ops.cc
 *
 * \brief Fuse Relay eligble sequences of Relay operators into a single one.
 *
 */
#include <tvm/relay/pass.h>
#include <tvm/runtime/module.h>
#include <tvm/lowered_func.h>
#include <tvm/operation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/logging.h>
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

using namespace runtime;

struct AbstractFusableOps : ExprMutator {
  Environment env;
  Array<GlobalVar> fusable_funcs;
  int counter = 0;
  size_t expr_hash;

  AbstractFusableOps(Environment env, size_t expr_hash) : env(env), expr_hash(expr_hash) {}

  Expr VisitExpr_(const CallNode* call) {
    if (auto op_node = call->op.as<OpNode>()) {
      // Placeholder fusion algorithm which abstracts
      // single definitions into functions only.
      Array<Var> params;
      Array<Expr> inner_args;
      Array<Expr> args;

      int param_number = 0;
      for (auto arg : call->args) {
        auto name = std::string("p") + std::to_string(param_number);
        auto type = arg->checked_type();
        auto var = VarNode::make(name, type);
        params.push_back(var);
        inner_args.push_back(var);
        args.push_back(VisitExpr(arg));
      }

      auto body = CallNode::make(call->op, inner_args, call->attrs);
      auto func = FunctionNode::make(params, body, call->checked_type(), {});
      func = func->SetAttr("Primitive", tvm::Integer(1));
      std::string func_name = "fused_";
      func_name += op_node->name;
      func_name += "_";
      func_name += std::to_string(counter++);
      func_name += "_";
      func_name += std::to_string(expr_hash);
      auto gv = GlobalVarNode::make(func_name);
      env->Add(gv, func);
      fusable_funcs.push_back(gv);
      return CallNode::make(gv, args, Attrs());
    } else {
      return ExprMutator::VisitExpr_(call);
    }
  }
};

Expr FuseOps(const Environment& env, const Expr& e) {
  // First we convert all chains of fusable ops into
  // abstracted functions which we mark as primtive
  // then we convert these primtive functions into
  // new operators.
  auto abstract = AbstractFusableOps(env, StructuralHash(e));
  auto abstracted_e = abstract.VisitExpr(e);
  RELAY_LOG(INFO) << "FuseOps: before=" << e
                  << "Fuse: after=" << abstracted_e;
  return abstracted_e;
}

TVM_REGISTER_API("relay._ir_pass.FuseOps")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = FuseOps(args[1], args[0]);
});


}  // namespace relay
}  // namespace tvm
