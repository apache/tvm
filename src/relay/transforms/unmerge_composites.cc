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
 * \file src/relay/transforms/unmerge_composites.cc
 * \brief Unmerges composite functions
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../analysis/call_graph.h"
#include "../op/call/call.h"

using namespace tvm::runtime;

namespace tvm {

namespace relay {

class Unmerger : ExprMutator {
 public:
  explicit Unmerger(CallGraphEntry* cur_node, CallGraphNode* call_graph)
      : cur_node_(cur_node), call_graph_(call_graph) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    // We can work with calls in both pre- and post-lowered form.
    Call vanilla_call = GetAnyCall(call_node);
    VLOG(1) << "Vanilla call " << vanilla_call->op << std::endl;
    VLOG(1) << "Vanilla call " << vanilla_call->op->checked_type_ << std::endl;
    // VLOG(1) << "Vanilla attrs " << vanilla_call->attrs << std::endl;
    const auto* global_var_node = vanilla_call->op.as<GlobalVarNode>();
    const auto* function_var_node = vanilla_call->op.as<FunctionNode>();
    // const auto* function__node = vanilla_call->op.as<Function>();

    if (global_var_node) {
      VLOG(1) << "Global Var node";
    }

    if (function_var_node) {
      VLOG(1) << "For existing function ";
      Function gv = GetRef<Function>(function_var_node);
      const auto* fn = gv.as<FunctionNode>();
      ICHECK(fn) << "Expected to work on a Relay function.";
      // auto func = Function(fn->params, fn->body, fn->ret_type, fn->type_params, fn->attrs);

      Array<Expr> new_args;
      new_args.reserve(vanilla_call->args.size());
      for (auto arg : vanilla_call->args) {
        new_args.push_back(VisitExpr(arg));
      }

      Map<Var, Expr> bind_map;
      for (size_t i = 0; i < new_args.size(); i++) {
        bind_map.Set(fn->params[i], new_args[i]);
      }

      auto func = Function(fn->params, fn->body, fn->ret_type, fn->type_params, {});
      VLOG(1) << "Params :" << func->params;
      VLOG(1) << "Ret type :" << func->ret_type;
      VLOG(1) << "Type Params :" << func->type_params;
      VLOG(1) << "Attrs :" << func->attrs;
      VLOG(1) << "Func body:" << func->body;

      return Bind(func->body, bind_map);

      // return func;
      // return func->body;
      // VLOG(1) << "gv " << func;
      // return func->body;
      // auto base_func = call_graph_->GetGlobalFunction(global);
    }
    // ICHECK(function_var_node);
    // VLOG(1) << std::endl << "function : " << function__node << std::endl;
    // VLOG(1) << "Function var node : " << function_var_node->attrs;
    // if(function_var_node) {
    //   VLOG(1) << "As function var node";
    // }
    // if (global_var_node) {
    // ICHECK(function_var_node);
    // ICHECK(global_var_node);
    // GlobalVar gv = GetRef<GlobalVar>(global_var_node);
    // Function gv = GetRef<Function>(function_var_node);

    // auto* cg_node = (*call_graph_)[gv->name_hint];
    // // if (CanInline(cg_node)) {
    // Array<Expr> new_args;
    // new_args.reserve(vanilla_call->args.size());
    // for (auto arg : vanilla_call->args) {
    //   new_args.push_back(VisitExpr(arg));
    // }
    // cur_node_->RemoveCallTo(gv);
    // return MakeNewExpr(gv, new_args, GetRef<Call>(call_node));
    // }
    // else: fallthrough
    // }
    // else: fallthrough

    // If not calling a global function then nothing to inline.
    return ExprMutator::VisitExpr_(call_node);
  }

  Expr VisitExpr_(const GlobalVarNode* gvn) final {
    GlobalVar gv = GetRef<GlobalVar>(gvn);
    auto* cg_node = (*call_graph_)[gv->name_hint];
    if (CanInline(cg_node)) {
      cur_node_->RemoveCallTo(gv);
      return MakeNewExpr(gv, {}, GetRef<GlobalVar>(gvn));
    }
    return ExprMutator::VisitExpr_(gvn);
  }

  Function Unmerge(const Function& func) {
    return WithFields(func, func->params, VisitExpr(func->body));
  }

 private:
  bool CanInline(const CallGraphEntry* cg_node) {
    // The node must be a leaf node and it cannot be recursive.
    return true;
    if (!cg_node->empty() || cg_node->IsRecursive()) return false;

    auto base_func = call_graph_->GetGlobalFunction(cg_node->GetGlobalVar());
    const auto* function_node = base_func.as<FunctionNode>();
    if (!function_node) {
      // Can't inline PrimFuncs!
      return false;
    }
    // The body of a global functions must be defined.
    if (!function_node->body.defined()) return false;

    // The function must be annotated with the inline attribute.
    // (Note that external functions do not have this attribute!)
    // if (!function_node->HasNonzeroAttr(attr::kInline)) return false;
    if (!function_node->HasNonzeroAttr(attr::kInline)) return false;

    // The function is not able to be inlined if any callee under the CallGraph
    // of this function cannot be inlined.
    for (const auto& it : *cg_node) {
      if (!CanInline(it.second)) {
        return false;
      }
    }

    return true;
  }

  // Make a new Relay expression to replace \p expr.
  Expr MakeNewExpr(const GlobalVar& global, const Array<Expr>& args, const Expr& expr) {
    ICHECK(expr->IsInstance<CallNode>() || expr->IsInstance<GlobalVarNode>());
    auto base_func = call_graph_->GetGlobalFunction(global);
    const auto* fn = base_func.as<FunctionNode>();
    VLOG(1) << "Make new expr " << fn;
    ICHECK(fn) << "Expected to work on a Relay function.";

    // There is an inconsistency here, the function itself gets shallow-copied but the body is not
    // shallow-copied.
    auto func = Function(fn->params, fn->body, fn->ret_type, fn->type_params, fn->attrs);
    // Inline the function body to the caller if this function uses default
    // compiler, i.e. no external codegen is needed.
    if (!func->GetAttr<String>(attr::kCompiler).defined() &&
        !func->GetAttr<String>(attr::kExternalSymbol).defined()) {
      ICHECK_EQ(func->params.size(), args.size())
          << "Mismatch found in the number of parameters and call args";
      // Bind the parameters with call args.
      Map<Var, Expr> bind_map;
      for (size_t i = 0; i < args.size(); i++) {
        bind_map.Set(fn->params[i], args[i]);
      }
      if (const auto* gvn = expr.as<GlobalVarNode>()) {
        auto ret_type = gvn->checked_type();
        // Cannot replace TensorType/TensorTupleType with FuncType. Therefore,
        // we simply inline the function as a closure instead of directly using
        // its body when the global var returns FuncType.
        return ret_type->IsInstance<FuncTypeNode>() ? std::move(func) : func->body;
      } else {
        ICHECK(expr->IsInstance<CallNode>());
        return Bind(func->body, bind_map);
      }
    } else if (const auto* call_node = expr.as<CallNode>()) {
      return Call(func, args, call_node->attrs, call_node->type_args);
    } else {
      return std::move(func);
    }
  }

  /*!
   * \brief The current call graph entry that is being handled. Each entry
   * contains a global function.
   */
  CallGraphEntry* cur_node_;
  /*! \brief The call graph that is used for global function lookup. */
  const CallGraphNode* call_graph_;
};

IRModule UnmergeComposites(const IRModule& module) {
  CallGraph cg(module);
  auto topo = cg->TopologicalOrder();
  std::reverse(topo.begin(), topo.end());
  std::unordered_set<CallGraphEntry*> original_entry;
  VLOG_CONTEXT << "Unmerge Composite";

  VLOG(1) << "Topo size: " << topo.size();
  for (auto* it : topo) {
    auto base_func = module->Lookup(it->GetNameHint());
    if (it->GetNameHint() != "main") {
      // Check kSymbol that is the correct one
      // Check kCompiler that is the correct one
      if (const auto* fn = base_func.as<FunctionNode>()) {
        VLOG(1) << "Func name " << it->GetNameHint() << std::endl << "-------" << std::endl;
        auto func = GetRef<Function>(fn);
        auto new_func = Unmerger(it, cg.operator->()).Unmerge(func);

        cg->module->Update(it->GetGlobalVar(), new_func);
      }
    }
  }
  VLOG(1) << "Post unmerge module " << std::endl;
  VLOG(1) << module;
  VLOG(1) << "------------- " << std::endl;
  return module;
}

namespace transform {

Pass UnmergeComposites() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::UnmergeComposites(m); };
  return CreateModulePass(pass_func, 1, "UnmergeComposites", {});
}

TVM_REGISTER_GLOBAL("relay._transform.UnmergeComposites").set_body_typed(UnmergeComposites);

}  // namespace transform

}  // namespace relay

}  // namespace tvm