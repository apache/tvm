/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file src/tvm/relay/pass/lower_ops.cc
 *
 * \brief Lower a Relay program to set of TVM operators.
 *
 */
#include <tvm/lowered_func.h>
#include <tvm/operation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/pass.h>
#include <tvm/runtime/module.h>
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

using namespace runtime;

LoweredOp LoweredOpNode::make(Function func, LoweredFunc lowered_func) {
  auto node = make_node<LoweredOpNode>();
  node->func = func;
  node->lowered_func = lowered_func;
  return LoweredOp(node);
}

struct AbstractLocalFunctions : ExprMutator {
  Environment env;
  size_t expr_hash;
  int counter = 0;
  std::unordered_set<GlobalVar, NodeHash, NodeEqual> visited_funcs;
  explicit AbstractLocalFunctions(Environment env)
      : env(env), expr_hash(0), counter(0), visited_funcs() {}

  Expr Abstract(const Expr& e) {
    expr_hash = StructuralHash(e);
    return VisitExpr(e);
  }

  Expr VisitExpr_(const GlobalVarNode* gvar_node) final {
    auto gvar = GetRef<GlobalVar>(gvar_node);
    auto it = visited_funcs.find(gvar);
    if (it == visited_funcs.end()) {
      auto func = env->Lookup(gvar);
      visited_funcs.insert(gvar);
      auto new_func = FunctionNode::make(
        func->params,
        VisitExpr(func->body),
        func->ret_type,
        func->type_params,
        func->attrs);
      env->Update(gvar, new_func);
    }
    return gvar;
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    Function func = GetRef<Function>(func_node);
    auto free_vars = FreeVars(func);
    Array<Var> params;
    for (auto free_var : free_vars) {
      auto var = VarNode::make("free_var", free_var->checked_type());
      params.push_back(var);
    }
    std::string abs_func = "abstracted_func_";
    abs_func += std::to_string(counter++);
    abs_func += std::to_string(expr_hash);
    auto gv = GlobalVarNode::make(abs_func);
    auto lifted_func = FunctionNode::make(params, func, Type(), {}, {});
    env->Add(gv, lifted_func);
    Array<Expr> args;
    for (auto free_var : free_vars) {
      args.push_back(free_var);
    }
    return CallNode::make(gv, args, {});
  }
};

struct LiveFunctions : ExprVisitor {
  Environment env;
  explicit LiveFunctions(Environment env) : env(env), global_funcs() {}

  std::unordered_set<GlobalVar, NodeHash, NodeEqual> visited_funcs;
  std::unordered_set<GlobalVar, NodeHash, NodeEqual> global_funcs;

  void Live(const Expr& e) {
    CHECK(!e.as<FunctionNode>())
        << "functions should of been transformed away by previous pass";
    VisitExpr(e);
  }

  void VisitExpr_(const FunctionNode* func_node) {
    LOG(FATAL) << "functions should of been transformed away by previous pass";
  }

  void VisitExpr_(const GlobalVarNode* var_node) final {
    GlobalVar var = GetRef<GlobalVar>(var_node);
    auto it = visited_funcs.find(var);
    if (it == visited_funcs.end()) {
      auto func = env->Lookup(var);
      visited_funcs.insert(var);
      // The last pass has trasnformed functions of the form:
      //
      // let x = fn (p_1, ..., p_n) { ... };
      // ...
      //
      // into:
      //
      // def abs_f(fv_1, ..., fv_n) {
      //    return (fn (p_1...,p_N) { ... }; }
      // }
      //
      // let x = abs_f(fv_1, ... fv_n);
      //
      // The only other case we can handle is
      //
      // fn foo(...) { body }
      //
      // We just search through the body in this case.
      if (auto inner_func = func->body.as<FunctionNode>()) {
        return VisitExpr(inner_func->body);
      } else {
        return VisitExpr(func->body);
      }
    }
  }

  void VisitExpr_(const CallNode* call) final {
    RELAY_LOG(INFO) << "LiveOps: CallNode=" << GetRef<Call>(call);
    if (auto gv_node = call->op.as<GlobalVarNode>()) {
      GlobalVar gvar = GetRef<GlobalVar>(gv_node);
      Function func = env->Lookup(gvar);

      auto attr = func->GetAttr("Primitive");

      if (attr.defined() && Downcast<Integer>(attr)->value == 1) {
        global_funcs.insert(gvar);
      } else {
         VisitExpr(gvar);
      }

      // Finally we need to ensure to visit all the args no matter what.
      for (auto arg : call->args) {
        VisitExpr(arg);
      }
    } else {
      return ExprVisitor::VisitExpr_(call);
    }
  }
};

using FCompute = TypedPackedFunc<Array<Tensor>(
    const Attrs&, const Array<Tensor>&, Type, std::string)>;
using FSchedule = TypedPackedFunc<Schedule(const Array<Tensor>&, std::string)>;

/*! \brief Return the set of operators in their TVM format. */
Array<LoweredOp> LowerOps(const Environment& env, const Expr& e,
                          const std::string& target) {
  RELAY_LOG(INFO) << "LowerOps: e=" << e;
  auto flower_ptr = Registry::Get("relay.op.compiler._lower");
  CHECK(flower_ptr);
  PackedFunc flower = *flower_ptr;

  auto abstracted_e = AbstractLocalFunctions(env).Abstract(e);
  auto live_funcs = LiveFunctions(env);
  live_funcs.VisitExpr(abstracted_e);

  auto schedule_reg = Op::GetAttr<FSchedule>("FTVMSchedule");
  auto compute_reg = Op::GetAttr<FCompute>("FTVMCompute");

  Array<LoweredOp> lowered_funcs;

  for (auto func_name : live_funcs.global_funcs) {
    auto func = env->Lookup(func_name);
    auto call = Downcast<Call>(func->body);
    auto op_node = call->op.as<OpNode>();
    CHECK(op_node);
    auto op = GetRef<Op>(op_node);

    // RELAY_LOG(INFO) << "LowerOps: Lowering " << op->name;

    // CHECK(IsPrimitiveOp(op)) << "failed to lower "
    // << op->name << "can only lower primitve operations";

    Array<Tensor> inputs;
    std::string input_name = "in";
    int i = 0;
    for (auto type_arg : call->type_args) {
      auto tt = Downcast<TensorType>(type_arg);
      inputs.push_back(PlaceholderOpNode::make(input_name + std::to_string(i),
                                               tt->shape, tt->dtype)
                           .output(0));
      i++;
    }

    auto output_tt = op->op_type->ret_type;
    Array<Tensor> outputs =
        compute_reg[op](call->attrs, inputs, output_tt, target);
    auto schedule = schedule_reg[op](outputs, target);
    size_t hash = ExprHash()(func);
    LoweredFunc lf =
        flower(op->name + std::to_string(hash), schedule, inputs, outputs);
    func = func->SetAttr("LoweredFunc", lf);
    env->Add(func_name, func, true);
    lowered_funcs.push_back(LoweredOpNode::make(func, lf));
  }

  return lowered_funcs;
}

TVM_REGISTER_API("relay._ir_pass.LowerOps")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = LowerOps(args[0], args[1]);
});


}  // namespace relay
}  // namespace tvm
