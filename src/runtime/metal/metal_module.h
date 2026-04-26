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
 * \file metal_module.h
 * \brief Execution handling of Metal kernels
 */
#ifndef TVM_RUNTIME_METAL_METAL_MODULE_H_
#define TVM_RUNTIME_METAL_METAL_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include "../metadata.h"

namespace tvm {
namespace runtime {
/*! \brief Maximum number of GPU supported in MetalModule. */
static constexpr const int kMetalMaxNumDevice = 32;

/*!
 * \brief create a metal module from data.
 *
 * \param smap The map from name to each shader kernel (FFI-typed).
 * \param fmap The map function information map of each function.
 * \param fmt The format of the source, can be "metal" or "metallib"
 * \param source Optional, source file, concatenated for debug dump
 *
 * Dispatches through the FFI registry ("ffi.Module.create.metal").
 * Requires libtvm_runtime built with USE_METAL=ON to have registered the creator.
 */
inline ffi::Module MetalModuleCreate(ffi::Map<ffi::String, ffi::String> smap,
                                     ffi::Map<ffi::String, FunctionInfo> fmap, ffi::String fmt,
                                     ffi::String source) {
  static const auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.metal");
  TVM_FFI_CHECK(fcreate.has_value(), RuntimeError)
      << "ffi.Module.create.metal is not registered in runtime. "
      << "Link or load libtvm_runtime built with USE_METAL=ON.";
  return (*fcreate)(smap, fmap, fmt, source).cast<ffi::Module>();
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_METAL_METAL_MODULE_H_
