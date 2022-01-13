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
 * \file src/runtime/metadata_module.cc
 * \brief A wrapper for initializing imported modules using metadata. This
 * module is intended to be used by various runtime in the TVM stack, i.e.
 * graph executor, relay VM, AOT runtime, and various user defined runtimes. It
 * paves the way to separate the code and metedata, which makes compilation
 * and/or interpretation more convenient. In addition, the clear separation of
 * code and metadata significantly reduces the efforts for handling external
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
 * \brief The metadata module is designed to manage initialization of the
 * imported submodules.
 */
class MetadataModuleNode : public ModuleNode {
 public:
  MetadataModuleNode(const std::unordered_map<std::string, NDArray>& metadata,
                     const std::unordered_map<std::string, std::vector<std::string>>& sym_vars)
      : metadata_(metadata), sym_vars_(sym_vars) {
    // Only the related submodules are cached to reduce the number of runtime
    // symbol lookup for initialization. Otherwise, symbols/primitives in the
    // DSO module will also be cached but they never need to be initialized.
    for (const auto& it : sym_vars_) {
      initialized_[it.first] = false;
    }
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    // Initialize and memoize the module.
    // Usually, we have some warmup runs. The module initialization should be
    // done at this stage. Therefore, runtime overhead is not a concern.
    if (initialized_.count(name) && !initialized_.at(name)) {
      this->InitSubModule(name);
      initialized_[name] = true;
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

  const char* type_key() const { return "metadata"; }

  /*!
   * \brief Get the list of metadata that is required by the given module.
   * \param symbol The symbol that is being queried.
   * \return The list of needed NDArray.
   */
  Array<NDArray> GetRequiredMetadata(const std::string& symbol) {
    Array<NDArray> ret;
    ICHECK_GT(sym_vars_.count(symbol), 0U) << "No symbol is recorded for " << symbol;
    std::vector<std::string> vars = sym_vars_[symbol];
    for (const auto& it : vars) {
      ICHECK_GT(metadata_.count(it), 0U) << "Found not recorded constant variable: " << it;
      ret.push_back(metadata_[it]);
    }
    return ret;
  }

  /*!
   * \brief Initialize each imported module.
   * \param symobl The symbol used for initializing a module. It is also used
   * for runtime lookup.
   *
   * \note  A module could be like the following:
   *  MetadataModuleNode (contains all the metadata)
   *    - CSourceModule
   *    - JSON runtime module
   *
   *  The initializer iterates through the imported modules and intilizes the
   *  found module accordingly by passing the needed metadata into it.
   */
  void InitSubModule(const std::string& symbol) {
    PackedFunc init(nullptr);
    for (Module it : this->imports()) {
      // Get the initialization function from the imported modules.
      std::string init_name = "__init_" + symbol;
      init = it.GetFunction(init_name, false);
      if (init != nullptr) {
        auto md = GetRequiredMetadata(symbol);
        // Initialize the module with metadata.
        int ret = init(md);
        // Report the error if initialization is failed.
        ICHECK_EQ(ret, 0) << TVMGetLastError();
        break;
      }
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    std::vector<std::string> variables;
    std::vector<NDArray> metadata;
    for (const auto& it : metadata_) {
      String var_name = it.first;
      variables.push_back(var_name);
      metadata.push_back(it.second);
    }

    // Save all variables in the function.
    stream->Write(variables);
    // Save all constant data.
    uint64_t sz = static_cast<uint64_t>(metadata.size());
    stream->Write(sz);
    for (uint64_t i = 0; i < sz; i++) {
      metadata[i].Save(stream);
    }

    // Save the symbol to list of required constant variables mapping
    std::vector<std::string> symbols;
    std::vector<std::vector<std::string>> const_vars;
    for (const auto& it : sym_vars_) {
      symbols.push_back(it.first);
      const_vars.push_back(it.second);
    }

    stream->Write(symbols);
    sz = static_cast<uint64_t>(sym_vars_.size());
    stream->Write(sz);
    for (uint64_t i = 0; i < sz; i++) {
      stream->Write(const_vars[i]);
    }
  }

  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);

    // Load the variables.
    std::vector<std::string> variables;
    ICHECK(stream->Read(&variables)) << "Loading variables failed";
    uint64_t sz;
    ICHECK(stream->Read(&sz, sizeof(sz))) << "Loading metadata size failed";
    ICHECK_EQ(static_cast<size_t>(sz), variables.size())
        << "The number of variables and ndarray counts must match";
    // Load the list of ndarray.
    std::vector<NDArray> arrays;
    for (uint64_t i = 0; i < sz; i++) {
      NDArray temp;
      temp.Load(stream);
      arrays.push_back(temp);
    }

    std::unordered_map<std::string, NDArray> metadata;
    for (uint64_t i = 0; i < sz; i++) {
      ICHECK_EQ(metadata.count(variables[i]), 0U);
      metadata[variables[i]] = arrays[i];
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

    std::unordered_map<std::string, std::vector<std::string>> sym_vars;
    for (uint64_t i = 0; i < sz; i++) {
      sym_vars[symbols[i]] = const_vars[i];
    }

    auto n = make_object<MetadataModuleNode>(metadata, sym_vars);
    return Module(n);
  }

 private:
  /*!
   * \brief Record if a module is initialized. It is needed by imported
   * modules using execution engine.
   */
  std::unordered_map<std::string, bool> initialized_;
  /*! \brief Variable name to NDArray mapping. */
  std::unordered_map<std::string, NDArray> metadata_;
  /*! \brief Symbol name to required constant variables mapping. */
  std::unordered_map<std::string, std::vector<std::string>> sym_vars_;
};

Module MetadataModuleCreate(
    const std::unordered_map<std::string, NDArray>& metadata,
    const std::unordered_map<std::string, std::vector<std::string>>& sym_vars) {
  auto n = make_object<MetadataModuleNode>(metadata, sym_vars);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_metadata")
    .set_body_typed(MetadataModuleNode::LoadFromBinary);
}  // namespace runtime
}  // namespace tvm
