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
  virtual const std::vector<std::string> GetExternLibPaths(std::string id = "") const = 0;

  /*!
   * \brief Get the function prefix of this compiler.
   *
   * \return A string of the function name prefix in the library.
   */
  virtual const std::string GetPrefix() const = 0;

  /*!
   * \brief Build the shared library of external ops.
   *
   * \param expr The subgraph Relay expression to be executed using extern ops.
   *
   */
  virtual void Build(const Expr& expr) = 0;

  /*!
   * \brief Get a PackedFunc from module, which is a function ptr can be invoked
   * for execution given some parameters.
   *
   * \param name the name of the external function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   */
  virtual runtime::PackedFunc GetFunction(
      const std::string& name, const std::shared_ptr<ModuleNode>& sptr_to_self) override = 0;

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
   * \brief Split the encoded function name to tokens.
   *
   * \param the function name string.
   *
   * \return a vector of tokenized function name splitted by "_".
   */
  std::string GetSubgraphID(Function& func) {
    const auto name_node = FunctionGetAttr(func, "func_name").as<tvm::ir::StringImm>();
    CHECK(name_node != nullptr) << "Fail to retrieve subgraph name.";
    std::string name = name_node->value;
    return GetSubgraphID(name);
  }
  
  std::string GetSubgraphID(std::string name) {
    std::string temp = name;
    std::vector<std::string> tokens;
    std::string delimiter = "_";
    size_t pos = 0;
    std::string token;
    while ((pos = temp.find(delimiter)) != std::string::npos) {
      token = temp.substr(0, pos);
      tokens.push_back(token);
      temp.erase(0, pos + delimiter.length());
    }
    tokens.push_back(temp);

    CHECK(tokens.size() >= 2) << "Invalid subgraph name: " << name;
    CHECK(tokens[0] == "subgraph") << "Function name does not start with \"subgraph\": " << name;
    return tokens[1];
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
    Close();
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
    auto sym = dlsym(handle_, name.c_str());
    char* error = dlerror();
    if (error) {
      CHECK(0) << "Fail to get symbol " << name << ": " << error;
    }
    return sym;
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
