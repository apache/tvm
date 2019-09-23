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

template <typename T>
using F2ARGS = void (*)(T* a, T* b);
template <typename T>
using F3ARGS = void (*)(T* a, T* b, T* c);
template <typename T>
using F4ARGS = void (*)(T* a, T* b, T* c, T* d);
template <typename T>
using F5ARGS = void (*)(T* a, T* b, T* c, T* d, T* e);
template <typename T>
using F6ARGS = void (*)(T* a, T* b, T* c, T* d, T* e, T* f);
template <typename T>
using F7ARGS = void (*)(T* a, T* b, T* c, T* d, T* e, T* f, T* g);
template <typename T>
using F8ARGS = void (*)(T* a, T* b, T* c, T* d, T* e, T* f, T* g, T* h);

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

  void SetSubgraphInfo(std::string id, const DLDataType type, int num_args) {
    subgraph_info_[id] = std::make_pair(type, num_args);
  }

  std::pair<DLDataType, int> GetSubgraphInfo(std::string id) {
    if (subgraph_info_.count(id) == 0) {
      LOG(FATAL) << "Info of subgraph " << id << " is missing.";
    }
    return subgraph_info_[id];
  }

  template<typename T>
  void Invoke(void* func_sym, std::vector<T*> data) {
    try {
      if (data.size() == 2) {
        auto func = reinterpret_cast<F2ARGS<T>>(func_sym);
        (*func)(data[0], data[1]);
      } else if (data.size() == 3) {
        auto func = reinterpret_cast<F3ARGS<T>>(func_sym);
        (*func)(data[0], data[1], data[2]);
      } else if (data.size() == 4) {
        auto func = reinterpret_cast<F4ARGS<T>>(func_sym);
        (*func)(data[0], data[1], data[2], data[3]);
      } else if (data.size() == 5) {
        auto func = reinterpret_cast<F5ARGS<T>>(func_sym);
        (*func)(data[0], data[1], data[2], data[3], data[4]);
      } else if (data.size() == 6) {
        auto func = reinterpret_cast<F6ARGS<T>>(func_sym);
        (*func)(data[0], data[1], data[2], data[3], data[4], data[5]);
      } else if (data.size() == 7) {
        auto func = reinterpret_cast<F7ARGS<T>>(func_sym);
        (*func)(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
      } else if (data.size() == 8) {
        auto func = reinterpret_cast<F8ARGS<T>>(func_sym);
        (*func)(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
      } else {
          LOG(FATAL) << "Unsupported argument number: " << data.size();
      }
    } catch (const std::exception& e) {
      LOG(FATAL) << "Execution failure: " << e.what();
    }
  }

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
    curr_id_ = GetSubgraphID(name);
    Open(this->GetExternLibPaths(curr_id_));
    CHECK(handle_) << "The external module has not been built or failed to open.\n";

    // Generate an external packed function
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      const DLTensor* dptr = ((runtime::NDArray) args[0]).operator->();

      // Check type and argument number
      auto info = GetSubgraphInfo(curr_id_);
      CHECK(info.first.code == dptr->dtype.code && info.first.bits == dptr->dtype.bits)
            << "Data type of subgraph " << curr_id_ << " and input is mismatch";
      CHECK(info.second == args.size())
            << "Argument number of subgraph " << curr_id_
            << " and input data is mismatch: " << info.second
            << " vs. " << args.size();

      // Get function from the library
      std::string encoded_name = GetPrefix() + curr_id_;
      auto func_sym = GetSymbol(encoded_name);

      // Reinterpret data and function to the right type and invoke
      if (runtime::TypeMatch(dptr->dtype, kDLFloat, 32)) {
        std::vector<float *> data;
        for (int i = 0; i < args.size(); ++i) {
          runtime::NDArray arg = args[i];
          data.push_back(reinterpret_cast<float*>(arg->data));
        }
        Invoke<float>(func_sym, data);
      } else if (runtime::TypeMatch(dptr->dtype, kDLFloat, 64)) {
        std::vector<double*> data;
        for (int i = 0; i < args.size(); ++i) {
          runtime::NDArray arg = args[i];
          data.push_back(reinterpret_cast<double*>(arg->data));
        }
        Invoke<double>(func_sym, data);
      } else {
        LOG(FATAL) << "Only support float32 and float64 types.";
      }
      //*rv = out;
    });
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

  bool IsOp(const CallNode* call, std::string op_name) {
    const auto* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "Expects a single op.";
    Op op = GetRef<Op>(op_node);
    return op == Op::Get(op_name);
  }

 protected:
  std::vector<int> GetShape(const Expr& expr) const {
    const auto* ttype = expr->checked_type().as<TensorTypeNode>();
    CHECK(ttype);
    std::vector<int> _shape;
    for (int i = 0; i < ttype->shape.size(); ++i) {
      auto* val = ttype->shape[i].as<IntImm>();
      CHECK(val);
      _shape.push_back(val->value);
    }
    return _shape;
  }

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

private:
  std::string curr_id_;
  std::unordered_map<std::string, std::pair<DLDataType, int>> subgraph_info_;
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_CONTRIB_CODEGEN_H_
