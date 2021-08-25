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
#include "let_list.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {

Expr ToBasicBlockNormalFormAux(const Expr& e) {
  // calculate all the dependency between nodes.
  support::Arena arena;
  DependencyGraph dg = DependencyGraph::Create(&arena, e);
  /* The scope of the whole expr is global.
   * The scope of any subexpr, is the lowest common ancestor of all incoming edge.
   * We also record the set of expressions whose scope is lifted.
   */
  std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg);
  return Fill::ToBasicBlockNormalForm(e, dg, &scopes.first, &scopes.second);
}

IRModule ToBasicBlockNormalForm(const IRModule& mod) {
  DLOG(INFO) << "ToBBlock:" << std::endl << mod;

  // Create a new module by shallow copy.
  auto mod_ = IRModule(mod->functions, mod->type_definitions, mod->Imports(), mod->source_map);

  tvm::Map<GlobalVar, Function> updates;
  auto funcs = mod_->functions;
  for (const auto& it : funcs) {
    ICHECK_EQ(FreeVars(it.second).size(), 0) << "Expected no free variables";
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
    }
    Expr ret = TransformF([&](const Expr& e) { return ToBasicBlockNormalFormAux(e); }, it.second);
    updates.Set(it.first, Downcast<Function>(ret));
  }

  for (auto pair : updates) {
    mod_->Add(pair.first, pair.second, true);
  }

  DLOG(INFO) << "ToBBlock: transformed" << std::endl << mod_;

  return mod_;
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
