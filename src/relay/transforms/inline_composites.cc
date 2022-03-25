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
 * \file src/relay/transforms/inline_composites.cc
 * \brief Undo the partioned graphs originate from merge composite.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../analysis/call_graph.h"
#include "../op/call/call.h"

using namespace tvm::runtime;

namespace tvm {

namespace relay {

class CompositeInliner : public MixedModeMutator {
 public:
  explicit CompositeInliner(CallGraphEntry* cur_node, CallGraphNode* call_graph)
      : cur_node_(cur_node), call_graph_(call_graph) {}

  Expr Rewrite_(const CallNode* call_node) {
    Call vanilla_call = GetAnyCall(call_node);
    const auto* function_node = vanilla_call->op.as<FunctionNode>();

    if (function_node) {
      Array<Expr> new_args;
      new_args.reserve(vanilla_call->args.size());
      for (auto arg : vanilla_call->args) {
        new_args.push_back(VisitExpr(arg));
      }

      Map<Var, Expr> bind_map;
      for (size_t i = 0; i < new_args.size(); i++) {
        bind_map.Set(function_node->params[i], new_args[i]);
      }

      // Attrs need to be empty at this point to avoid propagating Composite and
      // PartitionedFromPattern that fiddling TRT code gen for registered ops.
      return Bind(function_node->body, bind_map);
    }

    return MixedModeMutator::VisitExpr_(call_node);
  }

  Function Inline(const Function& func) {
    return WithFields(func, func->params, VisitExpr(func->body));
  }

 private:
  /*!
   * \brief The current call graph entry that is being handled. Each entry
   * contains a global function.
   */
  CallGraphEntry* cur_node_;
  /*! \brief The call graph that is used for global function lookup. */
  const CallGraphNode* call_graph_;
};

IRModule InlineComposites(const IRModule& module, runtime::String target) {
  CallGraph cg(module);
  auto topo = cg->TopologicalOrder();
  std::reverse(topo.begin(), topo.end());
  std::unordered_set<CallGraphEntry*> original_entry;
  ICHECK(target.defined());
  for (auto* it : topo) {
    auto base_func = module->Lookup(it->GetNameHint());

    if (!base_func->GetAttr<String>(attr::kCompiler).defined() &&
        base_func->GetAttr<String>(attr::kCompiler) != target) {
      continue;
    }

    if (it->GetNameHint() != "main") {
      if (const auto* fn = base_func.as<FunctionNode>()) {
        auto func = GetRef<Function>(fn);
        auto new_func = CompositeInliner(it, cg.operator->()).Inline(func);
        cg->module->Update(it->GetGlobalVar(), new_func);
      }
    }
  }
  return module;
}

namespace transform {

Pass InlineComposites(runtime::String target) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::InlineComposites(m, target); };
  return CreateModulePass(pass_func, 0, "InlineComposites", {});
}

TVM_REGISTER_GLOBAL("relay._transform.InlineComposites").set_body_typed(InlineComposites);

}  // namespace transform

}  // namespace relay

}  // namespace tvm
