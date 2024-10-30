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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/dead_code_elimination.cc
 * \brief Dead code elimination pass.
 * \sa tvm/relax/ir/binding_rewrite.cc
 *
 * Currently it removes:
 *   1. Unused local VarBindings
 *      (those where the bound var is unused and no impure operation is used).
 *   2. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 *
 * Any binding blocks that are left empty will be removed by the normalizer.
 */

#include <tvm/ir/analysis.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

IRModule RemoveUnusedFunctions(IRModule mod, const std::unordered_set<GlobalVar>& entry_funcs) {
  auto call_map = ir::CollectCallMap(mod);

  std::unordered_set<GlobalVar> reachable = entry_funcs;
  std::vector<GlobalVar> to_visit(entry_funcs.begin(), entry_funcs.end());
  bool all_callees_in_module = true;

  while (to_visit.size()) {
    GlobalVar visiting = to_visit.back();
    to_visit.pop_back();

    if (auto it = call_map.find(visiting); it != call_map.end()) {
      for (GlobalVar callee : (*it).second) {
        if (!reachable.count(callee)) {
          reachable.insert(callee);
          to_visit.push_back(callee);
        }
      }
    } else {
      all_callees_in_module = false;
    }
  }

  if (!all_callees_in_module) {
    return mod;
  }

  std::vector<GlobalVar> to_remove;
  for (const auto& [gvar, func] : mod->functions) {
    // The tracer contains all user-provided entry functions, all
    // externally-callable functions, and anything that is directly or
    // indirectly accessible from an entry function.
    if (!reachable.count(gvar)) {
      to_remove.push_back(gvar);
    }
  }

  if (to_remove.size()) {
    auto write_ptr = mod.CopyOnWrite();
    for (const auto& gvar : to_remove) {
      write_ptr->Remove(gvar);
    }
  }

  return mod;
}

IRModule DeadCodeElimination(const IRModule& arg_mod, Array<runtime::String> entry_function_names) {
  IRModule mod = arg_mod;

  // S0: Make a list of all user-specified entry functions and
  // externally-visible entry functions.
  std::unordered_set<GlobalVar> entry_functions;
  for (const auto& name : entry_function_names) {
    entry_functions.insert(mod->GetGlobalVar(name));
  }
  for (const auto& gv : mod->GetGlobalVars()) {
    const auto& func = mod->Lookup(gv);
    if (func.as<ExternFuncNode>() || func->GetLinkageType() == LinkageType::kExternal) {
      entry_functions.insert(gv);
    }
  }

  // S1: remove unused functions to reduce the number of functions to be analyzed.
  mod = RemoveUnusedFunctions(mod, entry_functions);

  // S2: remove unused variables in each function.
  {
    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<Function>()) {
        auto new_func = Downcast<Function>(RemoveAllUnused(opt.value()));
        if (!new_func.same_as(base_func)) {
          updates->Add(gvar, new_func);
        }
      }
    }
    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
  }

  // S3: remove unused functions again as some callers may be removed in S2.
  mod = RemoveUnusedFunctions(mod, entry_functions);

  return mod;
}

namespace transform {

Pass DeadCodeElimination(Array<runtime::String> entry_functions) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::DeadCodeElimination(m, entry_functions); };
  return CreateModulePass(pass_func, 1, "DeadCodeElimination", {});
}

TVM_REGISTER_GLOBAL("relax.transform.DeadCodeElimination").set_body_typed(DeadCodeElimination);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
