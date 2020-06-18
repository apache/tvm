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

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/support/logging.h>
#include <tvm/relay/transform.h>
#include <string>
#include <unordered_set>

#include "../analysis/call_graph.h"

using namespace tvm::runtime;

namespace tvm {
namespace relay {

class Inliner : ExprMutator {
 public:
  explicit Inliner(CallGraphEntry* cur_node, CallGraphNode* call_graph)
      : cur_node_(cur_node), call_graph_(call_graph) {}

  Expr VisitExpr_(const CallNode* call_node) final {
    Expr op = call_node->op;
    const auto* gvn = op.as<GlobalVarNode>();

    if (gvn) {
      GlobalVar gv = GetRef<GlobalVar>(gvn);
      auto* cg_node = (*call_graph_)[gv->name_hint];
      if (CanInline(cg_node)) {
        tvm::Array<Expr> call_args;
        for (auto arg : call_node->args) {
          auto new_arg = VisitExpr(arg);
          call_args.push_back(new_arg);
        }
        cur_node_->RemoveCallTo(gv);
        return MakeNewExpr(gv, call_args, GetRef<Call>(call_node));
      }
    }
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
    return Function(func->params,
                              VisitExpr(func->body),
                              func->ret_type,
                              func->type_params,
                              func->attrs);
  }

 private:
  bool CanInline(const CallGraphEntry* cg_node) {
    // The node must be a leaf node and it cannot be recursive.
    if (!cg_node->empty() || cg_node->IsRecursive()) return false;

    auto base_func = call_graph_->GetGlobalFunction(cg_node->GetGlobalVar());
    auto func = Downcast<Function>(base_func);
    // The body of a global functions must be defined.
    if (!func->body.defined()) return false;

    // The function must be annotated with the inline attribute.
    if (!func->HasNonzeroAttr(attr::kInline)) return false;

    // The function is not abled to be inlined if any callee under the CallGraph
    // of this function cannot be inlined.
    for (const auto& it : *cg_node) {
      if (!CanInline(it.second)) {
        return false;
      }
    }

    return true;
  }

  // Make a new Relay expression to replace the callee.
  Expr MakeNewExpr(const GlobalVar& global,
                   const Array<Expr>& args,
                   const Expr& callee) {
    CHECK(callee->IsInstance<CallNode>() ||
          callee->IsInstance<GlobalVarNode>());
    auto base_func = call_graph_->GetGlobalFunction(global);
    const auto* fn = base_func.as<FunctionNode>();
    CHECK(fn) << "Expected to work on a Relay function.";

    auto func = Function(fn->params,
                         fn->body,
                         fn->ret_type,
                         fn->type_params,
                         fn->attrs);
    // Inline the function body to the caller if this function uses default
    // compiler, i.e. no external codegen is needed.
    if (!func->GetAttr<String>(attr::kCompiler).defined()) {
      CHECK_EQ(func->params.size(), args.size())
          << "Mismatch found in the number of parameters and call args";
      // Bind the parameters with call args.
      Map<Var, Expr> bind_map;
      for (size_t i = 0; i < args.size(); i++) {
        bind_map.Set(fn->params[i], args[i]);
      }
      if (const auto* gvn = callee.as<GlobalVarNode>()) {
        auto ret_type = gvn->checked_type();
        // Cannot replace TensorType/TensorTupleType with FuncType. Therefore,
        // we simply inline the function as a closure instead of directly using
        // its body when the global var returns FuncType.
        return ret_type->IsInstance<FuncTypeNode>() ? std::move(func)
                                                    : func->body;
      } else {
        CHECK(callee->IsInstance<CallNode>());
        return Bind(func->body, bind_map);
      }
    } else if (const auto* call_node = callee.as<CallNode>()) {
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
    if (const auto* fn = base_func.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      auto new_func = Inliner(it, cg.operator->()).Inline(func);
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
    if (const auto* fn = base_func.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      if (func->HasNonzeroAttr(attr::kInline)) {
        CHECK_EQ(cgn->GetRefCount(), 0U)
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
    [=](IRModule m, PassContext pc) {
      return relay::Inline(m);
  };
  return CreateModulePass(pass_func, 1, "InlineGlobals", {});
}

TVM_REGISTER_GLOBAL("relay._transform.Inline")
.set_body_typed(Inline);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
