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

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

#include <algorithm>
#include <optional>
#include <tuple>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {

template <typename T>
using PSet = std::unordered_set<T, ObjectPtrHash, ObjectPtrEqual>;

template <typename T, typename U>
using PMap = std::unordered_map<T, U, ObjectPtrHash, ObjectPtrEqual>;

/* \brief Describes the modifications to be made for a function */
struct CalleeAnalysis {
  /* \brief The updated private function */
  Function func;

  /* \brief A function that updates the callsite arguments
   *
   * \param The arguments used to call the original function
   *
   * \return The arguments to be used for the modified function
   */
  std::function<Array<Expr>(Array<Expr>)> arg_updater;
};

std::optional<CalleeAnalysis> AnalyzeCallee(Function func) {
  bool is_exposed = func->attrs.GetAttr<String>(tvm::attr::kGlobalSymbol).defined();
  if (is_exposed) return std::nullopt;

  auto free_relax_vars = [&]() -> PSet<Var> {
    auto array_free_vars = FreeVars(func->body);
    return {array_free_vars.begin(), array_free_vars.end()};
  }();

  std::vector<bool> parameter_mask;
  parameter_mask.reserve(func->params.size());

  Array<Var> params;
  for (const auto& param : func->params) {
    bool is_used = free_relax_vars.count(param);
    parameter_mask.push_back(is_used);
    if (is_used) {
      params.push_back(param);
    }
  }

  if (func->params.size() == params.size()) {
    // Early bail-out for the common case where the function uses all
    // of its parameters.
    return std::nullopt;
  }

  // Even if a parameter is unused, it may provide definitions for
  // symbolic variables.  We still want to remove the relax variable
  // to reduce computational steps in the parent, but we need to
  // provide the symbolic variables the other steps.
  auto defined_tir_params = [&]() -> PSet<tir::Var> {
    auto param_sinfo =
        TupleStructInfo(params.Map([](const auto& var) { return GetStructInfo(var); }));
    auto arr = DefinableTIRVarsInStructInfo(param_sinfo);
    return {arr.begin(), arr.end()};
  }();

  // Use an array to define the order of the symbolic variables
  Array<tir::Var> free_tir_vars;
  for (const auto& tir_var : FreeSymbolicVars(func->body)) {
    if (!defined_tir_params.count(tir_var)) {
      free_tir_vars.push_back(tir_var);
    }
  }

  for (const auto& tir_var : free_tir_vars) {
    Var relax_var("param_" + tir_var->name_hint, PrimStructInfo(tir_var));
    params.push_back(relax_var);
  }

  FuncStructInfo new_sinfo(params.Map([](const auto& var) { return GetStructInfo(var); }),
                           func->ret_struct_info,
                           Downcast<FuncStructInfo>(func->struct_info_)->purity);

  auto arg_updater = [parameter_mask, old_relax_params = func->params,
                      free_tir_vars](Array<Expr> old_args) -> Array<Expr> {
    ICHECK_EQ(old_args.size(), parameter_mask.size())
        << "Call provides " << old_args.size() << ", but the callee accepts "
        << parameter_mask.size() << " parameters";

    Array<Expr> new_args;
    for (size_t i = 0; i < old_args.size(); i++) {
      if (parameter_mask.at(i)) {
        new_args.push_back(old_args[i]);
      }
    }

    if (free_tir_vars.size()) {
      Map<Var, Expr> old_binding;
      for (size_t i = 0; i < old_relax_params.size(); i++) {
        old_binding.Set(old_relax_params[i], old_args[i]);
      }
      arith::Analyzer analyzer;
      auto tir_binding = InferSymbolicVarMap(old_binding, &analyzer);

      for (const auto& tir_var : free_tir_vars) {
        new_args.push_back(PrimValue(tir_binding.at(tir_var)));
      }
    }

    return new_args;
  };

  auto write_ptr = func.CopyOnWrite();
  write_ptr->params = params;
  write_ptr->struct_info_ = new_sinfo;

  return CalleeAnalysis{func, arg_updater};
}

class CallSiteMutator : public ExprMutator {
 public:
  explicit CallSiteMutator(PMap<GlobalVar, std::function<Call(Call)>> callsite_updaters)
      : callsite_updaters_(callsite_updaters) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionNode* op) override {
    auto node = ExprMutator::VisitExpr_(op);

    // If a function was modified, that means it called into a private
    // function that now takes a reduced number of arguments.  Some
    // bindings in the calling scope, previously used to define those
    // unused arguments, may be able to be removed as a result.
    if (node.get() != op) {
      node = RemoveAllUnused(node);
    }
    return node;
  }

  Expr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(ExprMutator::VisitExpr_(op));

    if (auto gvar = node->op.as<GlobalVar>()) {
      if (auto it = callsite_updaters_.find(gvar.value()); it != callsite_updaters_.end()) {
        node = it->second(std::move(node));
      }
    }

    return node;
  }

  PMap<GlobalVar, std::function<Call(Call)>> callsite_updaters_;
};

}  // namespace

namespace transform {

Pass RemoveUnusedParameters() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) -> IRModule {
    PMap<GlobalVar, std::function<Call(Call)>> callsite_updaters;

    {
      IRModule new_callees;

      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto func = base_func.as<Function>()) {
          if (auto callee_res = AnalyzeCallee(func.value())) {
            auto new_func = callee_res->func;
            GlobalVar new_gvar(gvar->name_hint, new_func->checked_type_);
            new_gvar->struct_info_ = new_func->struct_info_;
            new_callees->Add(new_gvar, new_func);

            callsite_updaters[gvar] = [old_gvar = gvar, new_gvar,
                                       arg_updater = callee_res->arg_updater](Call call) -> Call {
              ICHECK(call->op.same_as(old_gvar)) << "InternalError: "
                                                 << "Updater should be applied to " << old_gvar
                                                 << ", but was applied to " << call->op;
              auto write_ptr = call.CopyOnWrite();
              write_ptr->op = new_gvar;
              write_ptr->args = arg_updater(call->args);
              return call;
            };
          }
        }
      }

      if (callsite_updaters.empty()) {
        return mod;
      }
      auto write_ptr = mod.CopyOnWrite();

      // Remove any private subroutines that have unused parameters,
      // then add the updated versions.  The new private functions
      // have the same name, but require a new GlobalVar to hold the
      // updated StructInfo.  As a result, calling `Update()` without
      // first calling `Remove()` introduce a duplicate name and
      // produce an error.
      for (const auto& it : callsite_updaters) {
        write_ptr->Remove(it.first);
      }
      write_ptr->Update(new_callees);
    }

    CallSiteMutator mutator(std::move(callsite_updaters));

    IRModule caller_updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto mutated = Downcast<Function>(mutator.VisitExpr(func.value()));
        if (!mutated.same_as(base_func)) {
          caller_updates->Add(gvar, mutated);
        }
      }
    }

    if (caller_updates->functions.size()) {
      mod.CopyOnWrite()->Update(caller_updates);
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "RemoveUnusedParameters", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RemoveUnusedParameters")
    .set_body_typed(RemoveUnusedParameters);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
