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
 *   1. Unused local VarBindings in a DataflowBlock.
 *   2. Unused DataflowBlocks in a function.
 *   3. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

/**
 * \brief Detects all the functions that can be possibly called by entry function.
 */
class CallTracer : public ExprVisitor {
 public:
  explicit CallTracer(IRModule mod) : mod_{mod}, called_funcs_{}, visiting_{} {}

  void VisitExpr_(const GlobalVarNode* op) final {
    called_funcs_.insert(GetRef<GlobalVar>(op));
    auto func = mod_->Lookup(op->name_hint);
    if (const auto* function_node = func.as<FunctionNode>()) {
      VisitExpr(GetRef<Function>(function_node));
    }
    // else: Don't visit PrimFuncs -- we don't need to collect any tir.Calls therein.
  }

  void VisitExpr_(const CallNode* call_node) final { ExprVisitor::VisitExpr_(call_node); }

  void VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);
    if (visiting_.find(func) == visiting_.end()) {
      visiting_.insert(func);
      for (auto param : func_node->params) {
        ExprVisitor::VisitExpr(param);
      }
      ExprVisitor::VisitExpr(func_node->body);
    }
  }

  void Trace(std::string entry) {
    called_funcs_.insert(mod_->GetGlobalVar(entry));
    auto main_func = mod_->Lookup(entry);
    VisitExpr(main_func);
  }

  bool check_if_called(GlobalVar gv) { return called_funcs_.count(gv) > 0; }

 private:
  IRModule mod_;

  // Record the names of all encountered functions.
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> called_funcs_;

  // Record the expressions that are being visited.
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visiting_;
};

IRModule RemoveUnusedFunctions(
    IRModule mod, const std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual>& entry_funcs) {
  CallTracer tracer(mod);
  for (const auto& gvar : entry_funcs) {
    tracer.VisitExpr(gvar);
  }

  std::vector<GlobalVar> to_remove;
  for (const auto& kv : mod->functions) {
    // The tracer contains all user-provided entry functions, all
    // externally-callable functions, and anything that is directly or
    // indirectly accessible from an entry function.
    if (!tracer.check_if_called(kv.first)) {
      to_remove.push_back(kv.first);
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
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> entry_functions;
  for (const auto& name : entry_function_names) {
    entry_functions.insert(mod->GetGlobalVar(name));
  }
  for (const auto& [gv, func] : mod->functions) {
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
