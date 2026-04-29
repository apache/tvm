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
 * \file primfunc_utils.cc
 * \brief Passes that serve as helper functions.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {
namespace transform {

transform::Pass AnnotateEntryFunc() {
  auto fpass = [](IRModule mod, transform::PassContext ctx) -> IRModule {
    // If only a single function exists, that function must be the entry
    if (mod->functions.size() == 1) {
      auto [gvar, base_func] = *mod->functions.begin();
      if (!base_func->HasNonzeroAttr(tirx::attr::kIsEntryFunc)) {
        if (auto ptr = base_func.as<PrimFuncNode>()) {
          mod->Update(gvar, WithAttr(ffi::GetRef<PrimFunc>(ptr), tirx::attr::kIsEntryFunc, true));
        }
      }
      return mod;
    }

    // If the module has multiple functions, but only one is exposed
    // externally, that function must be the entry.
    bool has_external_non_primfuncs = false;
    IRModule with_annotations;
    for (const auto& [gvar, base_func] : mod->functions) {
      bool is_external = base_func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).has_value();
      if (is_external) {
        if (auto ptr = base_func.as<PrimFuncNode>()) {
          with_annotations->Add(
              gvar, WithAttr(ffi::GetRef<PrimFunc>(ptr), tirx::attr::kIsEntryFunc, true));
        } else {
          has_external_non_primfuncs = true;
        }
      }
    }
    if (with_annotations->functions.size() == 1 && !has_external_non_primfuncs) {
      mod->Update(with_annotations);
      return mod;
    }

    // Default fallback, no annotations may be inferred.
    return mod;
  };
  return tvm::transform::CreateModulePass(fpass, 0, "tirx.AnnotateEntryFunc", {});
}

transform::Pass Filter(ffi::TypedFunction<bool(PrimFunc)> fcond) {
  auto fpass = [fcond](tirx::PrimFunc f, IRModule m, transform::PassContext ctx) {
    if (fcond(f)) {
      return f;
    } else {
      return tirx::PrimFunc(nullptr);
    }
  };
  return tirx::transform::CreatePrimFuncPass(fpass, 0, "tirx.Filter", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.transform.AnnotateEntryFunc", AnnotateEntryFunc)
      .def("tirx.transform.Filter", Filter);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
