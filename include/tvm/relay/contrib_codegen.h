/* * Licensed to the Apache Software Foundation (ASF) under one
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
#ifndef TVM_RELAY_CONTRIB_CODEGEN_H_
#define TVM_RELAY_CONTRIB_CODEGEN_H_

#include <dlpack/dlpack.h>
#include <stdlib.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/util.h>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tvm {
namespace relay {
namespace contrib {

class ExternModuleNodeBase : public runtime:: ModuleNode {
 public:
  ExternModuleNodeBase() = default;
  ~ExternModuleNodeBase() {
    Close();
  }

  /*!
   * \brief Get the full path of compiled external shared libraries of this compiler.
   *
   * \return An array of strings of the library paths.
   */
  virtual const std::vector<std::string> GetExternLibPaths() const = 0;

  /*!
   * \brief Build the shared library of external ops.
   *
   * \param expr The subgraph Relay expression to be executed using extern ops.
   *
   */
  virtual void Build(const Expr& expr) = 0;

  /*!
   * \brief The extern module specific implementation of invoking pre-built functions.
   *
   * \param name the name of the external function.
   * \param func_s The function symbol retrieved from the external library.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  virtual runtime::PackedFunc InvokeExternFunc(const std::string& name,
                                               const std::shared_ptr<ModuleNode>& sptr_to_self) = 0;

  /*!
   * \brief Get a PackedFunc from module, which is a function ptr can be invoked
   * for execution given some parameters.
   *
   * \param name the name of the external function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  runtime::PackedFunc GetFunction(const std::string& name,
                                  const std::shared_ptr<ModuleNode>& sptr_to_self) override {
    if (!IsHandleOpen()) {
      Open(this->GetExternLibPaths());
    }
    CHECK(handle_) << "The external module has not been built or failed to open.\n";

    auto func_s = this->InvokeExternFunc(name, sptr_to_self);
    char* error = dlerror();
    if (error != NULL) {
      LOG(FATAL) << error;
      return PackedFunc();
    }
    return func_s;
  }

  /*!
   * \brief Get the source code of the external module.
   *
   * \param format The format of the source code.
   *
   * \return The source code of the external library module in the text form.
   */
  TVM_DLL std::string GetSource(const std::string& format = "") override {
    return "";
  }

  const char* type_key() const override {
    return "ExternModule";
  }

  /*!
   * \brief Check if the library is opened or not.
   *
   *
   * \return True if the library is already opened.
   */
  virtual bool IsHandleOpen() {
    return handle_ != nullptr;
  }

 protected:
  // Platform dependent handlers for opening system lib.
#if defined(_WIN32)
  // The handle.
  HMODULE handle_{nullptr};

  // Open the library
  virtual void Open(const std::string& name) {
    std::wstring wname(name.begin(), name.end());
    handle_ = LoadLibraryW(wname.c_str());
    CHECK(handle_ != nullptr) << "Failed to open the dynamic shared library " << name;
  }

  // Retrieve a symbol.
  virtual void* GetSymbol(const std::string& name) {
    return reinterpret_cast<void*>(GetProcAddress(handle_, (LPCSTR)name.c_str()));  // NOLINT(*)
  }

  // Close the handle.
  virtual void Close() {
    if (handle_) {
      FreeLibrary(handle_);
    }
  }
#else
  // The handle.
  void* handle_{nullptr};

  // load the library
  virtual void Open(const std::vector<std::string> lib_names) {
    CHECK(lib_names.size() == 1) << "Default library loader only loads one library. "
                                 << "Please override the loader if multiple libraries are used";
    handle_ = dlopen(lib_names[0].c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(handle_ != nullptr) << "Failed to open the dynamic shared library " << lib_names[0] << " "
                              << dlerror();
  }

  /*!
   * \brief Retrieve the pre-compiled function symbol from the opened library.
   *
   * \param name the name of the external function.
   *
   * \return The pointer to the external function.
   * \note Exceptions when loading the symbol can be retrieved by dlerror().
   */
  virtual void* GetSymbol(const std::string& name) {
    return dlsym(handle_, name.c_str());
  }

  virtual void Close() {
    if (handle_) {
      dlclose(handle_);
    }
  }
#endif
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_CONTRIB_CODEGEN_H_
