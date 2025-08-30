
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
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <unordered_set>
#include <vector>

#include "module_internal.h"

namespace tvm {
namespace ffi {

Optional<Function> ModuleObj::GetFunction(const String& name, bool query_imports) {
  if (auto opt_func = this->GetFunction(name)) {
    return opt_func;
  }
  if (query_imports) {
    for (const Any& import : imports_) {
      if (auto opt_func = import.cast<Module>()->GetFunction(name, query_imports)) {
        return *opt_func;
      }
    }
  }
  return std::nullopt;
}

Optional<String> ModuleObj::GetFunctionMetadata(const String& name, bool query_imports) {
  if (auto opt_metadata = this->GetFunctionMetadata(name)) {
    return opt_metadata;
  }
  if (query_imports) {
    for (const Any& import : imports_) {
      if (auto opt_metadata = import.cast<Module>()->GetFunctionMetadata(name, query_imports)) {
        return *opt_metadata;
      }
    }
  }
  return std::nullopt;
}

void ModuleObj::ImportModule(const Module& other) {
  std::unordered_set<const ModuleObj*> visited{other.operator->()};
  std::vector<const ModuleObj*> stack{other.operator->()};
  while (!stack.empty()) {
    const ModuleObj* n = stack.back();
    stack.pop_back();
    for (const Any& m : n->imports_) {
      const ModuleObj* next = m.cast<const ModuleObj*>();
      if (visited.count(next)) continue;
      visited.insert(next);
      stack.push_back(next);
    }
  }
  if (visited.count(this)) {
    TVM_FFI_THROW(RuntimeError) << "Cyclic dependency detected during import";
  }
  imports_.push_back(other);
}

void ModuleObj::ClearImports() { imports_.clear(); }

bool ModuleObj::ImplementsFunction(const String& name, bool query_imports) {
  if (this->ImplementsFunction(name)) {
    return true;
  }
  if (query_imports) {
    for (const Any& import : imports_) {
      if (import.cast<Module>()->ImplementsFunction(name, query_imports)) {
        return true;
      }
    }
  }
  return false;
}

Module Module::LoadFromFile(const String& file_name) {
  String format = [&file_name]() -> String {
    const char* data = file_name.data();
    for (size_t i = file_name.size(); i > 0; i--) {
      if (data[i - 1] == '.') {
        return String(data + i, file_name.size() - i);
      }
    }
    TVM_FFI_THROW(RuntimeError) << "Failed to get file format from " << file_name;
    TVM_FFI_UNREACHABLE();
  }();

  if (format == "dll" || format == "dylib" || format == "dso") {
    format = "so";
  }
  String loader_name = "ffi.Module.load_from_file." + format;
  const auto floader = tvm::ffi::Function::GetGlobal(loader_name);
  if (!floader.has_value()) {
    TVM_FFI_THROW(RuntimeError) << "Loader for `." << format << "` files is not registered,"
                                << " resolved to (" << loader_name << ") in the global registry."
                                << "Ensure that you have loaded the correct runtime code, and"
                                << "that you are on the correct hardware architecture.";
  }
  return (*floader)(file_name, format).cast<Module>();
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  ModuleObj::InternalUnsafe::RegisterReflection();

  refl::GlobalDef()
      .def("ffi.ModuleLoadFromFile", &Module::LoadFromFile)
      .def_method("ffi.ModuleImplementsFunction",
                  [](Module mod, String name, bool query_imports) {
                    return mod->ImplementsFunction(name, query_imports);
                  })
      .def_method("ffi.ModuleGetFunctionMetadata",
                  [](Module mod, String name, bool query_imports) {
                    return mod->GetFunctionMetadata(name, query_imports);
                  })
      .def_method("ffi.ModuleGetFunction",
                  [](Module mod, String name, bool query_imports) {
                    return mod->GetFunction(name, query_imports);
                  })
      .def_method("ffi.ModuleGetPropertyMask", &ModuleObj::GetPropertyMask)
      .def_method("ffi.ModuleInspectSource", &ModuleObj::InspectSource)
      .def_method("ffi.ModuleGetKind", [](const Module& mod) -> String { return mod->kind(); })
      .def_method("ffi.ModuleGetWriteFormats", &ModuleObj::GetWriteFormats)
      .def_method("ffi.ModuleWriteToFile", &ModuleObj::WriteToFile)
      .def_method("ffi.ModuleImportModule", &ModuleObj::ImportModule)
      .def_method("ffi.ModuleClearImports", &ModuleObj::ClearImports);
});
}  // namespace ffi
}  // namespace tvm

int TVMFFIEnvModLookupFromImports(TVMFFIObjectHandle library_ctx, const char* func_name,
                                  TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = tvm::ffi::ModuleObj::InternalUnsafe::GetFunctionFromImports(
      reinterpret_cast<tvm::ffi::ModuleObj*>(library_ctx), func_name);
  TVM_FFI_SAFE_CALL_END();
}
