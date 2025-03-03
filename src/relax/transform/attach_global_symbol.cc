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
 * \file src/relax/transform/attach_global_symbol.cc
 * \brief Attach global_symbol to Relax functions and TIR Primfuncs for codegen.
 */

#include <tvm/ir/module.h>
#include <tvm/ir/replace_global_vars.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {
namespace transform {

Pass AttachGlobalSymbol() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    String c_prefix = mod->GetAttr<String>(tvm::attr::kSystemLibPrefix).value_or("");
    IRModule updates;
    Map<GlobalVar, GlobalVar> gvar_updates;

    for (const auto& [gvar, func] : mod->functions) {
      Optional<String> old_name = func->GetAttr<String>(tvm::attr::kGlobalSymbol);

      // TODO(tvm-team): re-enable once fix relax integration part
      // if (old_name) continue;

      Optional<String> new_name;
      BaseFunc new_func;

      if (auto* prim_func = func.as<tir::PrimFuncNode>()) {
        new_name = c_prefix + gvar->name_hint;
        new_func = WithAttr(GetRef<tir::PrimFunc>(prim_func), tvm::attr::kGlobalSymbol, new_name);
      } else if (auto* relax_func = func.as<FunctionNode>()) {
        new_name = gvar->name_hint;
        new_func = WithAttr(GetRef<Function>(relax_func), tvm::attr::kGlobalSymbol, new_name);
      }

      if (new_name.defined() && (!old_name.defined() || old_name.value() != new_name.value())) {
        updates->Add(gvar, new_func);
        if (new_name.value() != gvar->name_hint) {
          GlobalVar new_gvar(new_name.value());
          if (auto sinfo = gvar->struct_info_.as<StructInfo>()) {
            UpdateStructInfo(new_gvar, sinfo.value());
          }

          gvar_updates.Set(gvar, new_gvar);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);

      if (gvar_updates.size()) {
        mod = tvm::transform::ReplaceGlobalVars(mod, gvar_updates);
      }
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "AttachGlobalSymbol", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AttachGlobalSymbol").set_body_typed(AttachGlobalSymbol);
}  // namespace transform
}  // namespace relax
}  // namespace tvm
