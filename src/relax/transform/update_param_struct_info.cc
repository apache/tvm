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
 * \file tvm/relax/transform/update_param_struct_info.cc
 * \brief Mutate IRModule to accept new parameters
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <regex>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {
class ParamStructInfoMutator : public ExprMutator {
 public:
  explicit ParamStructInfoMutator(TypedPackedFunc<Optional<StructInfo>(Var)> sinfo_func)
      : sinfo_func_(sinfo_func) {}

  using ExprMutator::VisitExpr_;
  using ExprMutator::VisitVarDef_;

  Var VisitVarDef_(const VarNode* op) override {
    Var var = ExprMutator::VisitVarDef_(op);
    if (!inside_expr_) {
      if (auto new_sinfo = sinfo_func_(var)) {
        return Var(var->vid, new_sinfo);
      }
    }
    return var;
  }

  Expr VisitExpr_(const SeqExprNode* op) override {
    bool cache = inside_expr_;
    inside_expr_ = true;
    auto ret = ExprMutator::VisitExpr_(op);
    inside_expr_ = cache;
    return ret;
  }

  TypedPackedFunc<Optional<StructInfo>(Var)> sinfo_func_;
  bool inside_expr_{false};
};
}  // namespace

namespace transform {
Pass UpdateParamStructInfo(TypedPackedFunc<Optional<StructInfo>(Var)> sinfo_func) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    ParamStructInfoMutator mutator(sinfo_func);

    std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> to_remove;
    std::unordered_map<GlobalVar, Function, ObjectPtrHash, ObjectPtrEqual> to_add;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto updated = Downcast<Function>(mutator(func.value()));
        if (!updated.same_as(base_func)) {
          GlobalVar new_gvar(gvar->name_hint);
          UpdateStructInfo(new_gvar, GetStructInfo(updated));
          to_add.insert({new_gvar, updated});
          to_remove.insert(gvar);
        }
      }
    }

    if (to_remove.size() || to_add.size()) {
      auto write_ptr = mod.CopyOnWrite();

      for (const auto& gvar : to_remove) {
        write_ptr->Remove(gvar);
      }
      for (const auto& [gvar, func] : to_add) {
        write_ptr->Add(gvar, func);
      }
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 1, "UpdateParamStructInfo", {});
}

TVM_REGISTER_GLOBAL("relax.transform.UpdateParamStructInfo").set_body_typed(UpdateParamStructInfo);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
