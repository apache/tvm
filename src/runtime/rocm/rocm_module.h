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
 * \file rocm_module.h
 * \brief Execution handling of ROCM kernels
 */
#ifndef TVM_RUNTIME_ROCM_ROCM_MODULE_H_
#define TVM_RUNTIME_ROCM_ROCM_MODULE_H_

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <memory>
#include <string>
#include <vector>

#include "../metadata.h"

namespace tvm {
namespace runtime {

/*! \brief Maximum number of GPU supported in ROCMModule */
static constexpr const int kMaxNumGPUs = 32;

/*!
 * \brief create a rocm module from data.
 *
 * \param data The module data, can be hsaco
 * \param fmt The format of the data, can be "hsaco"
 * \param fmap The map function information map of each function.
 * \param rocm_source Optional, rocm source file
 * \param assembly Optional, GCN assembly source
 *
 * Dispatches through the FFI registry ("ffi.Module.create.rocm").
 * Requires libtvm_runtime built with USE_ROCM=ON to have registered the creator.
 */
inline ffi::Module ROCMModuleCreate(ffi::String data, ffi::String fmt,
                                    ffi::Map<ffi::String, FunctionInfo> fmap,
                                    ffi::String rocm_source, ffi::String assembly) {
  static const auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.rocm");
  TVM_FFI_CHECK(fcreate.has_value(), RuntimeError)
      << "ffi.Module.create.rocm is not registered in runtime. "
      << "Link or load libtvm_runtime built with USE_ROCM=ON.";
  return (*fcreate)(data, fmt, fmap, rocm_source, assembly).cast<ffi::Module>();
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_ROCM_ROCM_MODULE_H_
