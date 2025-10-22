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
 * \brief A wrapper for initializing imported modules using constant Tensor. This
 * module is intended to be used by various runtime in the TVM stack, i.e.
 * graph executor, relax VM, AOT runtime, and various user defined runtimes. It
 * paves the way to separate the code and metedata, which makes compilation
 * and/or interpretation more convenient. In addition, the clear separation of
 * code and constants significantly reduces the efforts for handling external
 * codegen and runtimes.
 */
#include <dmlc/memory_io.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/tensor.h>

#include <cstdint>

namespace tvm {
namespace runtime {

/*!
 * \brief The const-loader module is designed to manage initialization of the
 * imported submodules for the C++ runtime.
 */
class ConstLoaderModuleObj : public ffi::ModuleObj {
 public:
  ConstLoaderModuleObj(
      const std::unordered_map<std::string, Tensor>& const_var_tensor,
      const std::unordered_map<std::string, std::vector<std::string>>& const_vars_by_symbol)
      : const_var_tensor_(const_var_tensor), const_vars_by_symbol_(const_vars_by_symbol) {
    VLOG(1) << "Creating ConstLoaderModule";
    // Only the related submodules are cached to reduce the number of runtime
    // symbol lookup for initialization. Otherwise, symbols/primitives in the
    // DSO module will also be cached but they never need to be initialized.
    for (const auto& kv : const_vars_by_symbol_) {
      for (const auto& var : kv.second) {
        VLOG(1) << "ConstLoaderModuleNode has constant '" << var << "' for function '" << kv.first
                << "'";
        ICHECK_GT(const_var_tensor_.count(var), 0)
            << "ConstLoaderModuleNode is missing entry for constant '" << var << "' for function '"
            << kv.first << "'";
      }
      initialized_[kv.first] = false;
    }
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    VLOG(1) << "ConstLoaderModuleNode::GetFunction(" << name << ")";
    // Initialize and memoize the module.
    // Usually, we have some warmup runs. The module initialization should be
    // done at this stage. Therefore, runtime overhead is not a concern.
    if (initialized_.count(name) && !initialized_.at(name)) {
      this->InitSubModule(name);
      initialized_[name] = true;
    }
    ObjectRef _self = ffi::GetRef<ObjectRef>(this);

    if (name == "get_const_var_tensor") {
      return ffi::Function([_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        ffi::Map<ffi::String, ffi::Any> ret_map;
        for (const auto& kv : const_var_tensor_) {
          ret_map.Set(kv.first, kv.second);
        }
        *rv = ret_map;
      });
    }

    // Run the module.
    // Normally we would only have a limited number of submodules. The runtime
    // symobl lookup overhead should be minimal.
    ICHECK(!this->imports_.empty());
    for (const Any& it : this->imports_) {
      ffi::Optional<ffi::Function> pf = it.cast<ffi::Module>()->GetFunction(name);
      if (pf.has_value()) return pf.value();
    }
    return std::nullopt;
  }

  const char* kind() const final { return "const_loader"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; };

  /*!
   * \brief Get the list of constants that is required by the given module.
   * \param symbol The symbol that is being queried.
   * \return The list of needed Tensor.
   */
  ffi::Array<Tensor> GetRequiredConstants(const std::string& symbol) {
    ffi::Array<Tensor> ret;
    ICHECK_GT(const_vars_by_symbol_.count(symbol), 0U)
        << "No constants known for function '" << symbol << "'";
    std::vector<std::string> vars = const_vars_by_symbol_[symbol];
    for (const auto& var : vars) {
      ICHECK_GT(const_var_tensor_.count(var), 0U)
          << "No such constant variable '" << var << "' for function '" << symbol << "'";
      ret.push_back(const_var_tensor_[var]);
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
    for (const Any& it : this->imports_) {
      // Get the initialization function from the imported modules.
      std::string init_name = "__init_" + symbol;
      ffi::Optional<ffi::Function> init = it.cast<ffi::Module>()->GetFunction(init_name, false);
      if (init.has_value()) {
        auto md = GetRequiredConstants(symbol);
        // Initialize the module with constants.
        int ret = (*init)(md).cast<int>();
        // Report the error if initialization is failed.
        ICHECK_EQ(ret, 0);
        break;
      }
    }
  }

  ffi::Bytes SaveToBytes() const final {
    std::string bytes_buffer;
    dmlc::MemoryStringStream ms(&bytes_buffer);
    dmlc::Stream* stream = &ms;

    std::vector<std::string> variables;
    std::vector<Tensor> const_var_tensor;
    for (const auto& it : const_var_tensor_) {
      ffi::String var_name = it.first;
      variables.push_back(var_name);
      const_var_tensor.push_back(it.second);
    }

    // Save all variables in the function.
    stream->Write(variables);
    // Save all constant data.
    uint64_t sz = static_cast<uint64_t>(const_var_tensor.size());
    stream->Write(sz);
    for (uint64_t i = 0; i < sz; i++) {
      const_var_tensor[i].Save(stream);
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
    return ffi::Bytes(bytes_buffer);
  }

  static ffi::Module LoadFromBytes(const ffi::Bytes& bytes) {
    dmlc::MemoryFixedSizeStream ms(const_cast<char*>(bytes.data()), bytes.size());
    dmlc::Stream* stream = &ms;

    // Load the variables.
    std::vector<std::string> variables;
    ICHECK(stream->Read(&variables)) << "Loading variable names failed";
    uint64_t sz;
    ICHECK(stream->Read(&sz, sizeof(sz))) << "Loading number of vars failed";
    ICHECK_EQ(static_cast<size_t>(sz), variables.size())
        << "The number of variables and ndarray counts must match";
    // Load the list of ndarray.
    std::vector<Tensor> arrays;
    for (uint64_t i = 0; i < sz; i++) {
      Tensor temp;
      temp.Load(stream);
      arrays.push_back(temp);
    }

    std::unordered_map<std::string, Tensor> const_var_tensor;
    for (uint64_t i = 0; i < sz; i++) {
      ICHECK_EQ(const_var_tensor.count(variables[i]), 0U);
      const_var_tensor[variables[i]] = arrays[i];
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

    auto n = ffi::make_object<ConstLoaderModuleObj>(const_var_tensor, const_vars_by_symbol);
    return ffi::Module(n);
  }

 private:
  /*!
   * \brief Record if a module is initialized. It is needed by imported
   * modules using execution engine.
   */
  std::unordered_map<std::string, bool> initialized_;
  /*! \brief Variable name to Tensor mapping. */
  std::unordered_map<std::string, Tensor> const_var_tensor_;
  /*! \brief Symbol name to required constant variables mapping. */
  std::unordered_map<std::string, std::vector<std::string>> const_vars_by_symbol_;
};

ffi::Module ConstLoaderModuleCreate(
    const std::unordered_map<std::string, Tensor>& const_var_tensor,
    const std::unordered_map<std::string, std::vector<std::string>>& const_vars_by_symbol) {
  auto n = ffi::make_object<ConstLoaderModuleObj>(const_var_tensor, const_vars_by_symbol);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.Module.load_from_bytes.const_loader",
                        ConstLoaderModuleObj::LoadFromBytes);
}

}  // namespace runtime
}  // namespace tvm
