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
 *
 * \file to_basic_block_normal_form.cc
 *
 * \brief Turn an expression to the basic normal form.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include "../../support/arena.h"
#include "../analysis/dependency_graph.h"
#include "./pass_utils.h"

namespace tvm {
namespace relay {

IRModule ToBasicBlockNormalForm(const IRModule& mod) {
  // Create a new module by shallow copy.
  IRModule new_mod = mod->ShallowCopy();

  tvm::Map<GlobalVar, Function> updates;
  auto funcs = new_mod->functions;
  for (const auto& it : funcs) {
    ICHECK_EQ(FreeVars(it.second).size(), 0) << "Expected no free variables";
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);
      Function ret = Downcast<Function>(ToBasicBlockNormalFormAux(func));
      VLOG(1) << "rewritten:" << std::endl
              << PrettyPrint(func) << std::endl
              << "to BasicBlockANF:" << std::endl
              << PrettyPrint(ret);
      updates.Set(it.first, Downcast<Function>(ret));
    }
  }

  for (auto pair : updates) {
    new_mod->Add(pair.first, pair.second, true);
  }

  return new_mod;
}

bool BasicBlockNormalFormCheck(const Expr& e) {
  // calculate all the dependency between nodes.
  support::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, e);
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  for (auto expr : scopes.second) {
    LOG(FATAL) << "The expression below violates the basic block normal form in that "
               << "its scope should be lifted:\n"
               << expr;
  }
  return scopes.second.size() == 0;
}

TVM_REGISTER_GLOBAL("relay.analysis.check_basic_block_normal_form")
    .set_body_typed(BasicBlockNormalFormCheck);

namespace transform {

Pass ToBasicBlockNormalForm() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::ToBasicBlockNormalForm(m); };
  return CreateModulePass(pass_func, 1, "ToBasicBlockNormalForm", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ToBasicBlockNormalForm")
    .set_body_typed(ToBasicBlockNormalForm);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
