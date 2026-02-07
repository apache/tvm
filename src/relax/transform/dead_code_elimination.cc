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
 *   3. Unused function parameters
 *      We detect unused parameters after removing unused VarBindings and remove them from
 *      function signatures and call points
 *
 * Any binding blocks that are left empty will be removed by the normalizer.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
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

// two-stage dead parameter elimination
// 1. collect all unused parameters with propagation
// 2. update all call points with new functions and inputs
std::unordered_map<GlobalVar, std::vector<int>> CollectUsedParamIndices(const IRModule& mod) {
  std::unordered_map<GlobalVar, std::vector<int>> result;

  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto opt_func = base_func.as<Function>()) {
      auto func = opt_func.value();
      std::vector<bool> used(func->params.size(), false);

      PostOrderVisit(func->body, [&](const ObjectRef& obj) {
        if (auto v = obj.as<VarNode>()) {
          Var var = ffi::GetRef<Var>(v);
          for (size_t i = 0; i < func->params.size(); ++i) {
            if (var.same_as(func->params[i])) {
              used[i] = true;
            }
          }
        }
      });

      std::vector<int> indices;
      for (size_t i = 0; i < used.size(); ++i) {
        if (used[i]) indices.push_back(i);
      }

      result[gvar] = std::move(indices);
    }
  }

  return result;
}

struct CallSiteUpdater : public ExprMutator {
  const std::unordered_map<GlobalVar, std::vector<int>>& used_param_indices;

  explicit CallSiteUpdater(const std::unordered_map<GlobalVar, std::vector<int>>& used)
      : ExprMutator(std::nullopt), used_param_indices(used) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call) final {
    if (auto gvar = call->op.as<GlobalVar>()) {
      auto it = used_param_indices.find(gvar.value());
      if (it != used_param_indices.end()) {
        const auto& used = it->second;

        if (used.size() == call->args.size()) {
          return ExprMutator::VisitExpr_(call);
        }

        ffi::Array<Expr> new_args;
        for (int idx : used) {
          new_args.push_back(call->args[idx]);
        }

        auto new_call = Call(call->op, new_args, call->attrs);
        if (call->struct_info_.defined()) {
          new_call->struct_info_ = call->struct_info_;
        }
        return new_call;
      }
    }
    return ExprMutator::VisitExpr_(call);
  }
};

IRModule RemoveUnusedParameters(IRModule mod) {
  auto write_ptr = mod.CopyOnWrite();
  bool changed = true;

  do {
    changed = false;

    auto used_param_indices = CollectUsedParamIndices(mod);

    for (const auto& [gvar, used] : used_param_indices) {
      if (auto opt_func = mod->Lookup(gvar).as<Function>()) {
        auto func = opt_func.value();
        if (used.size() < func->params.size()) {
          changed = true;
          break;
        }
      }
    }

    if (!changed) break;

    std::vector<GlobalVar> worklist;
    std::unordered_set<GlobalVar> visited;
    std::function<void(GlobalVar)> dfs = [&](GlobalVar gvar) {
      if (visited.count(gvar)) return;
      visited.insert(gvar);

      if (auto opt_func = mod->Lookup(gvar).as<Function>()) {
        auto func = opt_func.value();
        PostOrderVisit(func->body, [&](const ObjectRef& obj) {
          if (auto call = obj.as<CallNode>()) {
            if (auto callee_gvar = call->op.as<GlobalVar>()) {
              dfs(callee_gvar.value());
            }
          }
        });
      }
      worklist.push_back(gvar);
    };

    for (const auto& [gvar, _] : mod->functions) {
      dfs(gvar);
    }

    for (const auto& gvar : worklist) {
      if (auto opt_func = mod->Lookup(gvar).as<Function>()) {
        auto func = opt_func.value();
        auto it = used_param_indices.find(gvar);
        if (it != used_param_indices.end() && it->second.size() < func->params.size()) {
          ffi::Array<Var> new_params;
          for (int old_idx : used_param_indices[gvar]) {
            new_params.push_back(func->params[old_idx]);
          }

          Function new_func = Function(new_params, func->body, func->ret_struct_info, func->is_pure,
                                       func->attrs, func->span);
          write_ptr->Update(gvar, new_func);
        }
      }
    }

    CallSiteUpdater updater(used_param_indices);
    for (const auto& gvar : worklist) {
      if (auto opt_func = mod->Lookup(gvar).as<Function>()) {
        auto func = opt_func.value();
        Expr updated_body = updater(func->body);

        if (!updated_body.same_as(func->body)) {
          Function new_func(func->params, updated_body, func->ret_struct_info, func->is_pure,
                            func->attrs, func->span);
          write_ptr->Update(gvar, new_func);
        }
      }
    }
  } while (changed);

  return mod;
}

IRModule DeadCodeElimination(const IRModule& arg_mod,
                             ffi::Array<ffi::String> entry_function_names) {
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

  // S3: remove unused parameters in each function
  mod = RemoveUnusedParameters(mod);

  // S4: remove unused functions again as some callers may be removed in S2 and S3.
  mod = RemoveUnusedFunctions(mod, entry_functions);

  // S5: remove unused variables again as some arguments may be removed in S3
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

  return mod;
}

namespace transform {

Pass DeadCodeElimination(ffi::Array<ffi::String> entry_functions) {
  auto pass_func = [=](IRModule m, PassContext pc) {
    return relax::DeadCodeElimination(m, entry_functions);
  };
  return CreateModulePass(pass_func, 1, "DeadCodeElimination", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.DeadCodeElimination", DeadCodeElimination);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
