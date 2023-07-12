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
#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {
namespace transform {

Pass AttachGlobalSymbol() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    mod.CopyOnWrite();

    String c_prefix = mod->GetAttr<String>(tvm::attr::kSystemLibPrefix).value_or("");
    std::vector<std::pair<GlobalVar, BaseFunc> > updates;

    for (auto& p : mod->functions) {
      BaseFunc func = p.second;
      // TODO(tvm-team): re-enable once fix relax integration part
      // if (func->GetAttr<String>(tvm::attr::kGlobalSymbol)) continue;
      if (auto* prim_func = func.as<tir::PrimFuncNode>()) {
        updates.emplace_back(p.first,
                             WithAttr(GetRef<tir::PrimFunc>(prim_func), tvm::attr::kGlobalSymbol,
                                      c_prefix + p.first->name_hint));
      } else if (auto* relax_func = func.as<FunctionNode>()) {
        updates.emplace_back(p.first, WithAttr(GetRef<Function>(relax_func),
                                               tvm::attr::kGlobalSymbol, p.first->name_hint));
      }
    }
    for (const auto& pair : updates) {
      mod->Add(pair.first, pair.second, true);
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "AttachGlobalSymbol", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AttachGlobalSymbol").set_body_typed(AttachGlobalSymbol);
}  // namespace transform
}  // namespace relax
}  // namespace tvm
