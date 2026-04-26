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
 * \file const_loader_module.h
 * \brief Defines an interface to use the ConstLoaderModule.
 */

#ifndef TVM_RUNTIME_CONST_LOADER_MODULE_H_
#define TVM_RUNTIME_CONST_LOADER_MODULE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/tensor.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Create a ConstLoader module object.
 *
 * \param const_var_tensor Maps const var name to Tensor containing data for the var.
 * \param const_vars_by_symbol Maps the name of a module init function to a list of names of
 * const vars whose data will be passed to that init function.
 *
 * \return The created ConstLoaderModule.
 *
 * Dispatches through the FFI registry ("ffi.Module.create.const_loader").
 * The creator is always available (ConstLoaderModule is a runtime-universal module).
 */
inline ffi::Module ConstLoaderModuleCreate(
    const std::unordered_map<std::string, Tensor>& const_var_tensor,
    const std::unordered_map<std::string, std::vector<std::string>>& const_vars_by_symbol) {
  static const auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.const_loader");
  TVM_FFI_CHECK(fcreate.has_value(), RuntimeError)
      << "ffi.Module.create.const_loader is not registered in runtime. "
      << "Ensure libtvm_runtime is loaded.";
  // Convert to FFI-compatible types.
  ffi::Map<ffi::String, Tensor> ffi_const_var_tensor;
  for (const auto& kv : const_var_tensor) {
    ffi_const_var_tensor.Set(kv.first, kv.second);
  }
  ffi::Map<ffi::String, ffi::Array<ffi::String>> ffi_const_vars_by_symbol;
  for (const auto& kv : const_vars_by_symbol) {
    ffi::Array<ffi::String> vars;
    for (const auto& v : kv.second) {
      vars.push_back(ffi::String(v));
    }
    ffi_const_vars_by_symbol.Set(kv.first, vars);
  }
  return (*fcreate)(ffi_const_var_tensor, ffi_const_vars_by_symbol).cast<ffi::Module>();
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONST_LOADER_MODULE_H_
