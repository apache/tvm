/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

Function FunctionCheckForSpecialCase(
    Function func,
    Map<Variant<tir::Var, relax::Var, String>, Variant<Expr, PrimExpr>> arg_special_case) {
  // Early bail-out if no updates need to be made.
  if (arg_special_case.empty()) {
    return func;
  }

  Array<tir::Var> old_symbolic_vars = DefinedSymbolicVars(func);

  // Map from string to the variable(s) with that name.
  std::unordered_map<std::string, Array<Variant<tir::Var, relax::Var>>> string_lookup;
  std::unordered_set<const tir::VarNode*> symbolic_var_set;
  std::unordered_set<const relax::VarNode*> relax_var_set;
  for (const auto& var : old_symbolic_vars) {
    string_lookup[var->name_hint].push_back(var);
    symbolic_var_set.insert(var.get());
  }
  for (const auto& param : func->params) {
    string_lookup[param->name_hint()].push_back(param);
    relax_var_set.insert(param.get());
  }

  auto get_tir_expr = [](const tir::Var& var,
                         const Variant<Expr, PrimExpr>& replacement) -> PrimExpr {
    if (auto prim = replacement.as<PrimExpr>()) {
      return prim.value();
    }

    if (auto prim_value = replacement.as<relax::PrimValue>()) {
      return prim_value.value()->value;
    }

    if (auto relax_expr = replacement.as<relax::ExprNode>()) {
      auto sinfo = relax_expr->struct_info_;
      auto prim_sinfo = sinfo.as<PrimStructInfoNode>();
      if (prim_sinfo && prim_sinfo->value.defined()) {
        return prim_sinfo->value.value();
      } else if (prim_sinfo) {
        LOG(FATAL) << "ValueError: "
                   << "Attempted to replace TIR variable " << var << " with " << replacement
                   << ".  While this is a relax.PrimValue, is is not bound to a known PrimExpr, "
                      "and cannot be used to specialize a TIR variable.  "
                   << "Consider defining the PrimValue with R.Prim(value=tir_var) instead of "
                      "R.Prim(dtype=dtype)";
      } else {
        LOG(FATAL) << "ValueError: "
                   << "Attempted to replace TIR variable " << var << " with " << replacement
                   << ", but relax expression of type " << relax_expr->GetTypeKey()
                   << " with struct info " << sinfo << " cannot be converted to a known PrimExpr.";
      }
    }

    LOG(FATAL) << "InternalError: "
               << "Variant did not contain one of the allowed types";
  };

  auto get_relax_expr = [](const Variant<Expr, PrimExpr>& replacement) -> Expr {
    if (auto relax_expr = replacement.as<Expr>()) {
      return relax_expr.value();
    } else if (auto tir_expr = replacement.as<PrimExpr>()) {
      return relax::PrimValue(tir_expr.value());
    } else {
      LOG(FATAL) << "InternalError: "
                 << "Variant did not contain one of the allowed types";
    }
  };

  // Replacement maps to be used when rewriting the function.
  Map<relax::Var, Expr> relax_remap;
  Map<tir::Var, PrimExpr> tir_remap;
  for (const auto& [key, replacement] : arg_special_case) {
    if (auto opt = key.as<String>()) {
      String string_key = opt.value();
      auto it = string_lookup.find(string_key);
      CHECK(it != string_lookup.end())
          << "The name \"" << string_key << "\" does not correspond to either "
          << "a parameter or a symbolic variable in the function signature.  "
          << "The function has parameters named "
          << func->params.Map([](const relax::Var& var) { return var->name_hint(); })
          << " and symbolic variables " << old_symbolic_vars;

      CHECK_EQ(it->second.size(), 1)
          << "The name \"" << string_key << "\" does not uniquely identify "
          << "a function parameter or symbolic variable.  "
          << "This name could refer to any of " << it->second.Map([](const auto& var) -> String {
               std::stringstream ss;
               ss << var << " (" << var->GetTypeKey() << ")";
               return ss.str();
             });

      auto var = it->second[0];

      if (auto opt = var.as<tir::Var>()) {
        auto tir_var = opt.value();
        CHECK(!tir_remap.count(tir_var))
            << "Remap of TIR variable " << tir_var << " was defined multiple times";
        tir_remap.Set(tir_var, get_tir_expr(tir_var, replacement));
      } else if (auto opt = var.as<relax::Var>()) {
        auto relax_var = opt.value();
        CHECK(!relax_remap.count(relax_var))
            << "Remap of Relax variable " << relax_var << " was defined multiple times";
        relax_remap.Set(relax_var, get_relax_expr(replacement));
      } else {
        LOG(FATAL) << "InternalError: "
                   << "Variant did not match any allowed type";
      }
    } else if (auto opt = key.as<tir::Var>()) {
      auto tir_var = opt.value();

      CHECK(!tir_remap.count(tir_var))
          << "Remap of symbolic variable " << tir_var << " was defined multiple times";
      CHECK(symbolic_var_set.count(tir_var.get()))
          << "ValueError: "
          << "Expected all tir::Var special cases to appear as symbolic variables "
          << "within the function signature.  "
          << "Function signature has symbolic variables " << old_symbolic_vars;

      tir_remap.Set(tir_var, get_tir_expr(tir_var, replacement));
    } else if (auto opt = key.as<relax::Var>()) {
      auto relax_var = opt.value();

      CHECK(!relax_remap.count(relax_var))
          << "Remap of variable " << relax_var << " was defined multiple times";
      CHECK(relax_var_set.count(relax_var.get()))
          << "ValueError: "
          << "Expected all relax::Var special cases to appear as function parameters, "
          << "but variable " << relax_var << " is not one of the function parameters.  "
          << "Function has parameters " << func->params;
    } else {
      LOG(FATAL) << "Expected symbolic variable to be a tir::Var or a string name, "
                 << "but " << key << " was of type " << key->GetTypeKey();
    }
  }

  // The condition for symbolic variable special cases is collected in
  // order of their occurrence in the function signature.  Generating
  // `tir_cond` by iterating over `tir_remap` would produce equivalent
  // expressions, but the order would be unspecified, which would make
  // it difficult to test.
  PrimExpr tir_cond = Bool(true);
  for (auto it = old_symbolic_vars.rbegin(); it != old_symbolic_vars.rend(); it++) {
    const auto& tir_var = *it;
    if (auto expr = tir_remap.Get(tir_var)) {
      tir_cond = tir_cond && (tir_var == expr.value());
    }
  }
  ICHECK(relax_remap.empty()) << "Not yet supported";

  Function general_case = func;
  Function special_case = Downcast<Function>(Bind(func, relax_remap, tir_remap));

  auto convert_to_lambda = [](Function func) -> Function {
    func = WithoutAttr(std::move(func), tvm::attr::kGlobalSymbol);
    func = CopyWithNewVars(std::move(func));
    return func;
  };
  special_case = convert_to_lambda(special_case);
  general_case = convert_to_lambda(general_case);

  {
    auto free_symbolic_vars = FreeSymbolicVars(special_case);
    CHECK(free_symbolic_vars.empty())
        << "Resulting special case should not have any undefined symbolic variables, "
        << "but TIR variables " << free_symbolic_vars << " were undefined.";
  }

  relax::Var general_case_var("general_case", GetStructInfo(general_case));
  relax::Var special_case_var("special_case", GetStructInfo(special_case));
  relax::Var output("output", func->ret_struct_info);

  Array<Binding> bindings = {
      VarBinding(general_case_var, general_case),
      VarBinding(special_case_var, special_case),
      VarBinding(output,
                 relax::If(relax::PrimValue(tir_cond),
                           relax::Call(special_case_var,
                                       func->params.Map([](Var var) -> Expr { return var; })),
                           relax::Call(general_case_var,
                                       func->params.Map([](Var var) -> Expr { return var; })))),
  };

  func.CopyOnWrite()->body = SeqExpr({BindingBlock(bindings)}, output);

  func = Normalize(func);

  return func;
}

TVM_REGISTER_GLOBAL("relax.FunctionCheckForSpecialCase")
    .set_body_typed(FunctionCheckForSpecialCase);

}  // namespace relax
}  // namespace tvm
