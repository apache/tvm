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
#include <tvm/relax/type.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relax {

Function FunctionBindSymbolicVars(Function func, Map<ObjectRef, PrimExpr> obj_remap) {
  Array<tir::Var> old_symbolic_vars = DefinedSymbolicVars(func);

  // Map from string to the variable(s) with that name.
  std::unordered_map<std::string, Array<tir::Var>> string_lookup;
  std::unordered_set<const tir::VarNode*> symbolic_var_set;
  for (const auto& var : old_symbolic_vars) {
    string_lookup[var->name_hint].push_back(var);
    symbolic_var_set.insert(var.get());
  }

  // Replacement map to be used when rewriting the function.
  Map<tir::Var, PrimExpr> var_remap;
  for (const auto& [key, replacement] : obj_remap) {
    if (auto opt = key.as<String>()) {
      String string_key = opt.value();
      auto it = string_lookup.find(string_key);
      CHECK(it != string_lookup.end())
          << "Function does not use symbolic var with name \"" << string_key << "\".  "
          << "Function has symbolic variables " << old_symbolic_vars;

      CHECK_EQ(it->second.size(), 1)
          << "Function contains multiple symbolic variables with name \"" << string_key << "\".  "
          << "The TIR variables " << it->second << " are all named \"" << string_key << "\"";
      auto var = it->second[0];

      CHECK(!var_remap.count(var)) << "Remap of variable " << var << " was defined multiple times";
      var_remap.Set(var, replacement);
    } else if (auto opt = key.as<tir::Var>()) {
      auto var = opt.value();

      CHECK(!var_remap.count(var)) << "Remap of variable " << var << " was defined multiple times";
      CHECK(symbolic_var_set.count(var.get()))
          << "Function does not use variable " << var << " as a symbolic variable.  "
          << "Function has symbolic variables " << old_symbolic_vars;
      var_remap.Set(var, replacement);
    } else {
      LOG(FATAL) << "Expected symbolic variable to be a tir::Var or a string name, "
                 << "but " << key << " was of type " << key->GetTypeKey();
    }
  }

  auto new_func = Downcast<Function>(Bind(func, {}, var_remap));

  auto free_symbolic_vars = FreeSymbolicVars(new_func);

  CHECK(free_symbolic_vars.empty())
      << "Resulting function should not have any undefined symbolic variables, "
      << "but TIR variables " << free_symbolic_vars << " were undefined.";

  return new_func;
}

TVM_REGISTER_GLOBAL("relax.FunctionBindSymbolicVars").set_body_typed(FunctionBindSymbolicVars);

}  // namespace relax
}  // namespace tvm
