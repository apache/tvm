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
 * \file dso_libary.cc
 * \brief Create library module to load from dynamic shared library.
 */
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "library_module.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tvm {
namespace runtime {

// Dynamic shared libary.
// This is the default module TVM used for host-side AOT
class DSOLibrary final : public Library {
 public:
  ~DSOLibrary() {
    if (lib_handle_) Unload();
  }
  void Init(const std::string& name) { Load(name); }

  void* GetSymbol(const char* name) final { return GetSymbol_(name); }

 private:
  // Platform dependent handling.
#if defined(_WIN32)
  // library handle
  HMODULE lib_handle_{nullptr};

  void* GetSymbol_(const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(lib_handle_, (LPCSTR)name));  // NOLINT(*)
  }

  // Load the library
  void Load(const std::string& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
    ICHECK(lib_handle_ != nullptr) << "Failed to load dynamic shared library " << name;
  }

  void Unload() {
    FreeLibrary(lib_handle_);
    lib_handle_ = nullptr;
  }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const std::string& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    ICHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name << " " << dlerror();
  }

  void* GetSymbol_(const char* name) { return dlsym(lib_handle_, name); }

  void Unload() {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
#endif
};

TVM_REGISTER_GLOBAL("runtime.module.loadfile_so").set_body([](TVMArgs args, TVMRetValue* rv) {
  auto n = make_object<DSOLibrary>();
  n->Init(args[0]);
  *rv = CreateModuleFromLibrary(n);
});
}  // namespace runtime
}  // namespace tvm
