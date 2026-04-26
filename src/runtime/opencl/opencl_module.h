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
 * \file opencl_module.h
 * \brief Execution handling of OPENCL kernels
 */
#ifndef TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
#define TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>

#include <string>
#include <unordered_map>

#include "../../support/bytes_io.h"
#include "../metadata.h"
#include "../vulkan/spirv_shader.h"

namespace tvm {
namespace runtime {
/*!
 * \brief Create a opencl module for GPU devices from data.
 *
 * \param data The module data.
 * \param fmt The format of the data, can be "clbin", "cl"
 * \param fmap The map function information map of each function.
 * \param source Generated OpenCL kernels.
 *
 * Dispatches through the FFI registry ("ffi.Module.create.opencl").
 * Requires libtvm_runtime built with USE_OPENCL=ON to have registered the creator.
 */
inline ffi::Module OpenCLModuleCreate(ffi::String data, ffi::String fmt,
                                      ffi::Map<ffi::String, FunctionInfo> fmap,
                                      ffi::String source) {
  static const auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.opencl");
  TVM_FFI_CHECK(fcreate.has_value(), RuntimeError)
      << "ffi.Module.create.opencl is not registered in runtime. "
      << "Link or load libtvm_runtime built with USE_OPENCL=ON.";
  return (*fcreate)(data, fmt, fmap, source).cast<ffi::Module>();
}

/*!
 * \brief Create a opencl module from SPIRV.
 *
 * \param shaders The map from function names to SPIRV binaries.
 * \param spirv_text The concatenated text representation of SPIRV modules.
 * \param fmap The map function information map of each function.
 *
 * Dispatches through the FFI registry ("ffi.Module.create.opencl.spirv").
 * Each SPIRVShader is serialised to ffi::Bytes before crossing the FFI boundary.
 * Requires libtvm_runtime built with USE_OPENCL=ON and TVM_ENABLE_SPIRV to have
 * registered the creator.
 */
inline ffi::Module OpenCLModuleCreate(
    const std::unordered_map<std::string, spirv::SPIRVShader>& shaders, ffi::String spirv_text,
    ffi::Map<ffi::String, FunctionInfo> fmap) {
  static const auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.opencl.spirv");
  TVM_FFI_CHECK(fcreate.has_value(), RuntimeError)
      << "ffi.Module.create.opencl.spirv is not registered in runtime. "
      << "Link or load libtvm_runtime built with USE_OPENCL=ON and TVM_ENABLE_SPIRV.";
  // Serialise each SPIRVShader to ffi::Bytes for the FFI boundary.
  ffi::Map<ffi::String, ffi::Bytes> shader_bytes;
  for (const auto& kv : shaders) {
    std::string buf;
    support::BytesOutStream strm(&buf);
    strm.Write(kv.second);
    shader_bytes.Set(kv.first, ffi::Bytes(std::move(buf)));
  }
  return (*fcreate)(shader_bytes, spirv_text, fmap).cast<ffi::Module>();
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
