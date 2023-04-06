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
 * \file src/relay/transforms/inline.cc
 * \brief Global function inliner. It contains the following steps:
 *
 *  - Preprocessing: eligibility checking. Only inline the functions that can
 *  be inlined. We currently only use simple rules to make the decision. No
 *  profitibility analysis is available for now.
 *
 *  - Inline: replace the call with a function or the function body depending on
 *  the attribute of the callee function. For example, we return the function
 *  node when it doesn't use default compiler, i.e. llvm. This is because these
 *  functions are packed to be offloaded to external codegen.
 *
 *  - Postprocessing: remove the replaced functions that have no reference.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <string>
#include <unordered_set>

#include "../analysis/call_graph.h"
#include "../op/call/call.h"

using namespace tvm::runtime;

namespace tvm {
namespace relay {

class Inliner : ExprMutator {
 public:
  explicit Inliner(CallGraphEntry* cur_node, CallGraphNode* call_graph)
      : cur_node_(cur_node), call_graph_(call_graph) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    // We can work with calls in both pre- and post-lowered form.
    Call vanilla_call = GetAnyCall(call_node);

    const auto* global_var_node = vanilla_call->op.as<GlobalVarNode>();
    if (global_var_node) {
      GlobalVar gv = GetRef<GlobalVar>(global_var_node);
      auto* cg_node = (*call_graph_)[gv->name_hint];
      if (CanInline(cg_node)) {
        Array<Expr> new_args;
        new_args.reserve(vanilla_call->args.size());
        for (auto arg : vanilla_call->args) {
          new_args.push_back(VisitExpr(arg));
        }
        // TODO(mbs): Does not handle multiple calls to the same global function.
        cur_node_->RemoveCallTo(gv);
        return MakeNewExpr(gv, new_args, GetRef<Call>(call_node));
      }
      // else: fallthrough
    }
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

  Function Inline(const Function& func) {
    return WithFields(func, func->params, VisitExpr(func->body));
  }

 private:
  bool CanInline(const CallGraphEntry* cg_node) {
    // The node must be a leaf node and it cannot be recursive.
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
    // (Note that partitioned functions and external functions do not have this attribute!)
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
    ICHECK(fn) << "Expected to work on a Relay function.";

    // There is an inconsistency here, the function itself gets shallow-copied but the body is not
    // shallow-copied.
    auto func = Function(fn->params, fn->body, fn->ret_type, fn->type_params, fn->attrs);
    // Inline the function body to the caller if this function uses default
    // compiler, i.e. no external codegen is needed.
    if (!func->GetAttr<String>(attr::kCompiler).defined() && !func->HasNonzeroAttr(attr::kExtern)) {
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

IRModule Inline(const IRModule& module) {
  CallGraph cg(module);
  auto topo = cg->TopologicalOrder();
  // Get the reverse topological order of the global functions.
  std::reverse(topo.begin(), topo.end());
  // Cache the functions that are originally entries. These functions will
  // remain in the module after inlining.
  std::unordered_set<CallGraphEntry*> original_entry;

  for (auto* it : topo) {
    if (it->GetRefCount() == 0) original_entry.emplace(it);
    // Skip the leaf calls and the recursive calls that don't call other
    // functions.
    if (it->empty() || (it->IsRecursive() && it->size() == 1)) continue;
    auto base_func = module->Lookup(it->GetNameHint());
    if (auto func = base_func.as<Function>()) {
      auto new_func = Inliner(it, cg.operator->()).Inline(func.value());
      // TODO(zhiics) Maybe move this to CallGraph, but updating function from
      // CallGraph arbitarily may lead to incorrect CallGraph.
      cg->module->Update(it->GetGlobalVar(), new_func);
    }
  }

  // Clean up the functions that are inlined and have no reference.
  for (auto* cgn : topo) {
    // Skip recursive functions and entry functions even if they are marked as
    // `inline`.
    if (cgn->IsRecursive() || original_entry.count(cgn)) continue;
    auto base_func = cg->GetGlobalFunction(cgn->GetGlobalVar());
    // Skip calls to PrimFuncs since they can't be inlined.
    if (const auto* func = base_func.as<FunctionNode>()) {
      if (func->HasNonzeroAttr(attr::kInline)) {
        ICHECK_EQ(cgn->GetRefCount(), 0U)
            << cgn->GetNameHint() << " is marked as inline but not inlined.";
        cgn->CleanCallGraphEntries();
        cg->RemoveGlobalVarFromModule(cgn, /*update_call_graph*/ true);
      }
    }
  }

  return cg->module;
}

namespace transform {

Pass Inline() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::Inline(m); };
  return CreateModulePass(pass_func, 1, "InlineGlobals", {});
}

TVM_REGISTER_GLOBAL("relay._transform.Inline").set_body_typed(Inline);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
