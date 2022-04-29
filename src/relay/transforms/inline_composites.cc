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
  CompositeInliner() = default;

  using MixedModeMutator::Rewrite_;

  Expr Rewrite_(const CallNode* call_node, const Expr& post) final {
    const auto* post_call_node = post.as<CallNode>();
    Call vanilla_post_call = GetAnyCall(post_call_node);
    if (const auto* function_node = vanilla_post_call->op.as<FunctionNode>()) {
      if (function_node->GetAttr(attr::kComposite, Optional<String>()).defined()) {
        // Is a call to a literal function with the "Composite" attribute.
        // Inline the function body.
        Map<Var, Expr> bind_map;
        for (size_t i = 0; i < vanilla_post_call->args.size(); i++) {
          bind_map.Set(function_node->params[i], vanilla_post_call->args[i]);
        }
        return Bind(function_node->body, bind_map);
      }
    }
    return post;
  }

  Function Inline(const Function& func) {
    return WithFields(func, /*opt_params=*/{}, VisitExpr(func->body));
  }
};

IRModule InlineComposites(const IRModule& module, runtime::String target) {
  IRModule out_mod = module->ShallowCopy();
  for (const auto& kv : module->functions) {
    Optional<String> opt_compiler = kv.second->GetAttr(attr::kCompiler, Optional<String>());
    if (const auto* function_node = kv.second.as<FunctionNode>()) {
      if (opt_compiler.defined() && opt_compiler.value() == target) {
        // Is a global function with the "Compiler" attribute matching the desired target.
        // Inline all "Composite" function calls in the body.
        out_mod->Add(kv.first, CompositeInliner().Inline(GetRef<Function>(function_node)));
      }
    }
  }
  return out_mod;
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
