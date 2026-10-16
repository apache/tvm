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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tirx {

const FunctionNode* FindEntryFunc(const IRModule& mod, GlobalVar* result_g_var) {
  ffi::Optional<GlobalVar> result, last_gvar;
  // Priority 1: Function marked as `tirx::attr::kIsEntryFunc`
  int num_function = 0;
  const tirx::FunctionNode* main_func = nullptr;
  const tirx::FunctionNode* last_func = nullptr;
  for (const auto& kv : mod->functions) {
    GlobalVar gv = kv.first;
    BaseFunc base_func = kv.second;
    if (const auto* func = base_func.as<tirx::FunctionNode>()) {
      last_func = func;
      last_gvar = gv;
      if (func->HasNonzeroAttr(tvm::tirx::attr::kIsEntryFunc)) {
        if (result_g_var != nullptr) {
          *result_g_var = gv;
        }
        return func;
      }
      if (gv->name_hint == "main") {
        main_func = func;
        result = gv;
      }
      ++num_function;
    }
  }
  // Priority 2: Function whose name is `main`
  if (main_func != nullptr) {
    if (result_g_var != nullptr) {
      *result_g_var = result.value();
    }
    return main_func;
  }
  // Priority 3: The only Function in the IRModule
  if (num_function == 1) {
    if (result_g_var != nullptr) {
      *result_g_var = last_gvar.value();
    }
    return last_func;
  }
  return nullptr;
}

}  // namespace tirx
}  // namespace tvm
