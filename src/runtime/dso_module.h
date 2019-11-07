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
 * \file src/runtime/dso_module.h
 * \brief Module to load from dynamic shared library.
 */
#ifndef TVM_RUNTIME_DSO_MODULE_H_
#define TVM_RUNTIME_DSO_MODULE_H_

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tvm {
namespace runtime {

// Module to load from dynamic shared libary.
// This is the default module TVM used for host-side AOT
class DSOModuleNode : public ModuleNode {
 public:
  ~DSOModuleNode() {
    if (lib_handle_) Unload();
  }

  virtual const char* type_key() const {
    return "dso";
  }

  /*!
   * \brief Get a PackedFunc from module, which is a function ptr can be invoked
   * for execution given some parameters. This function can be implemented by
   * different backends as well to implement their own way of retrieving
   * a function poninter and invoking it.
   *
   * \param name the name of the external function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) override;

  /*!
   * \brief Initialize the module using a prvided shared library.
   * \param name. The dynamically linked shared library.
   */
  void Init(const std::string& name);

 protected:
  // Platform dependent handling.
#if defined(_WIN32)
  // library handle
  HMODULE lib_handle_{nullptr};
  // Load the library
  void Load(const std::string& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name;
  }
  void* GetSymbol(const char* name) {
    return reinterpret_cast<void*>(
        GetProcAddress(lib_handle_, (LPCSTR)name)); // NOLINT(*)
  }
  void Unload() {
    FreeLibrary(lib_handle_);
  }
  // Check if the handle_ is open.
  bool IsLoaded() const {
    rewturn lib_handle_ != nullptr;
  }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const std::string& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name
        << " " << dlerror();
  }
  void* GetSymbol(const char* name) {
    return dlsym(lib_handle_, name);
  }
  void Unload() {
    dlclose(lib_handle_);
  }
  // Check if the handle_ is open.
  bool IsLoaded() const {
    return lib_handle_ != nullptr;
  }
#endif
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DSO_MODULE_H_
