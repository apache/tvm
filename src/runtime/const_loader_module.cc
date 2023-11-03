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
 * \file src/runtime/const_loader_module.cc
 * \brief A wrapper for initializing imported modules using constant NDArray. This
 * module is intended to be used by various runtime in the TVM stack, i.e.
 * graph executor, relay VM, AOT runtime, and various user defined runtimes. It
 * paves the way to separate the code and metedata, which makes compilation
 * and/or interpretation more convenient. In addition, the clear separation of
 * code and constants significantly reduces the efforts for handling external
 * codegen and runtimes.
 */
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <sstream>

#include "meta_data.h"

namespace tvm {
namespace runtime {

/*!
 * \brief The const-loader module is designed to manage initialization of the
 * imported submodules for the C++ runtime.
 */
class ConstLoaderModuleNode : public ModuleNode {
 public:
  ConstLoaderModuleNode(
      const std::unordered_map<std::string, NDArray>& const_var_ndarray,
      const std::unordered_map<std::string, std::vector<std::string>>& const_vars_by_symbol)
      : const_var_ndarray_(const_var_ndarray), const_vars_by_symbol_(const_vars_by_symbol) {
    VLOG(1) << "Creating ConstLoaderModule";
    // Only the related submodules are cached to reduce the number of runtime
    // symbol lookup for initialization. Otherwise, symbols/primitives in the
    // DSO module will also be cached but they never need to be initialized.
    for (const auto& kv : const_vars_by_symbol_) {
      for (const auto& var : kv.second) {
        VLOG(1) << "ConstLoaderModuleNode has constant '" << var << "' for function '" << kv.first
                << "'";
        ICHECK_GT(const_var_ndarray_.count(var), 0)
            << "ConstLoaderModuleNode is missing entry for constant '" << var << "' for function '"
            << kv.first << "'";
      }
      initialized_[kv.first] = false;
    }
  }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    VLOG(1) << "ConstLoaderModuleNode::GetFunction(" << name << ")";
    // Initialize and memoize the module.
    // Usually, we have some warmup runs. The module initialization should be
    // done at this stage. Therefore, runtime overhead is not a concern.
    if (initialized_.count(name) && !initialized_.at(name)) {
      this->InitSubModule(name);
      initialized_[name] = true;
    }

    if (name == "get_const_var_ndarray") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Map<String, ObjectRef> ret_map;
        for (const auto& kv : const_var_ndarray_) {
          ret_map.Set(kv.first, kv.second);
        }
        *rv = ret_map;
      });
    }

    // Run the module.
    // Normally we would only have a limited number of submodules. The runtime
    // symobl lookup overhead should be minimal.
    ICHECK(!this->imports().empty());
    for (Module it : this->imports()) {
      PackedFunc pf = it.GetFunction(name);
      if (pf != nullptr) return pf;
    }
    return PackedFunc(nullptr);
  }

  const char* type_key() const final { return "const_loader"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kBinarySerializable; };

  /*!
   * \brief Get the list of constants that is required by the given module.
   * \param symbol The symbol that is being queried.
   * \return The list of needed NDArray.
   */
  Array<NDArray> GetRequiredConstants(const std::string& symbol) {
    Array<NDArray> ret;
    ICHECK_GT(const_vars_by_symbol_.count(symbol), 0U)
        << "No constants known for function '" << symbol << "'";
    std::vector<std::string> vars = const_vars_by_symbol_[symbol];
    for (const auto& var : vars) {
      ICHECK_GT(const_var_ndarray_.count(var), 0U)
          << "No such constant variable '" << var << "' for function '" << symbol << "'";
      ret.push_back(const_var_ndarray_[var]);
    }
    return ret;
  }

  /*!
   * \brief Initialize each imported module.
   * \param symobl The symbol used for initializing a module. It is also used
   * for runtime lookup.
   *
   * \note  A module could be like the following:
   *  ConstLoaderModuleNode (contains all the constants)
   *    - CSourceModule
   *    - JSON runtime module
   *
   *  The initializer iterates through the imported modules and intilizes the
   *  found module accordingly by passing the needed constants into it.
   */
  void InitSubModule(const std::string& symbol) {
    PackedFunc init(nullptr);
    for (Module it : this->imports()) {
      // Get the initialization function from the imported modules.
      std::string init_name = "__init_" + symbol;
      init = it.GetFunction(init_name, false);
      if (init != nullptr) {
        auto md = GetRequiredConstants(symbol);
        // Initialize the module with constants.
        int ret = init(md);
        // Report the error if initialization is failed.
        ICHECK_EQ(ret, 0) << TVMGetLastError();
        break;
      }
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    std::vector<std::string> variables;
    std::vector<NDArray> const_var_ndarray;
    for (const auto& it : const_var_ndarray_) {
      String var_name = it.first;
      variables.push_back(var_name);
      const_var_ndarray.push_back(it.second);
    }

    // Save all variables in the function.
    stream->Write(variables);
    // Save all constant data.
    uint64_t sz = static_cast<uint64_t>(const_var_ndarray.size());
    stream->Write(sz);
    for (uint64_t i = 0; i < sz; i++) {
      const_var_ndarray[i].Save(stream);
    }

    // Save the symbol to list of required constant variables mapping
    std::vector<std::string> symbols;
    std::vector<std::vector<std::string>> const_vars;
    for (const auto& it : const_vars_by_symbol_) {
      symbols.push_back(it.first);
      const_vars.push_back(it.second);
    }

    stream->Write(symbols);
    sz = static_cast<uint64_t>(const_vars_by_symbol_.size());
    stream->Write(sz);
    for (uint64_t i = 0; i < sz; i++) {
      stream->Write(const_vars[i]);
    }
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);

    // Load the variables.
    std::vector<std::string> variables;
    ICHECK(stream->Read(&variables)) << "Loading variable names failed";
    uint64_t sz;
    ICHECK(stream->Read(&sz, sizeof(sz))) << "Loading number of vars failed";
    ICHECK_EQ(static_cast<size_t>(sz), variables.size())
        << "The number of variables and ndarray counts must match";
    // Load the list of ndarray.
    std::vector<NDArray> arrays;
    for (uint64_t i = 0; i < sz; i++) {
      NDArray temp;
      temp.Load(stream);
      arrays.push_back(temp);
    }

    std::unordered_map<std::string, NDArray> const_var_ndarray;
    for (uint64_t i = 0; i < sz; i++) {
      ICHECK_EQ(const_var_ndarray.count(variables[i]), 0U);
      const_var_ndarray[variables[i]] = arrays[i];
    }

    // Load the symbol to list of required constant variables mapping
    std::vector<std::string> symbols;
    ICHECK(stream->Read(&symbols)) << "Loading symbols failed";
    ICHECK(stream->Read(&sz, sizeof(sz))) << "Loading number of symbols failed";
    ICHECK_EQ(static_cast<size_t>(sz), symbols.size());
    std::vector<std::vector<std::string>> const_vars;
    for (uint64_t i = 0; i < sz; i++) {
      std::vector<std::string> vars;
      ICHECK(stream->Read(&vars)) << "Loading const variables failed";
      const_vars.push_back(vars);
    }

    std::unordered_map<std::string, std::vector<std::string>> const_vars_by_symbol;
    for (uint64_t i = 0; i < sz; i++) {
      const_vars_by_symbol[symbols[i]] = const_vars[i];
    }

    auto n = make_object<ConstLoaderModuleNode>(const_var_ndarray, const_vars_by_symbol);
    return Module(n);
  }

 private:
  /*!
   * \brief Record if a module is initialized. It is needed by imported
   * modules using execution engine.
   */
  std::unordered_map<std::string, bool> initialized_;
  /*! \brief Variable name to NDArray mapping. */
  std::unordered_map<std::string, NDArray> const_var_ndarray_;
  /*! \brief Symbol name to required constant variables mapping. */
  std::unordered_map<std::string, std::vector<std::string>> const_vars_by_symbol_;
};

Module ConstLoaderModuleCreate(
    const std::unordered_map<std::string, NDArray>& const_var_ndarray,
    const std::unordered_map<std::string, std::vector<std::string>>& const_vars_by_symbol) {
  auto n = make_object<ConstLoaderModuleNode>(const_var_ndarray, const_vars_by_symbol);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_metadata")
    .set_body_typed(ConstLoaderModuleNode::LoadFromBinary);
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_const_loader")
    .set_body_typed(ConstLoaderModuleNode::LoadFromBinary);

}  // namespace runtime
}  // namespace tvm
