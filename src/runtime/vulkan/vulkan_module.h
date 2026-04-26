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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_MODULE_H_
#define TVM_RUNTIME_VULKAN_VULKAN_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>

#include <string>
#include <unordered_map>

#include "../../support/bytes_io.h"
#include "../metadata.h"
#include "spirv_shader.h"

namespace tvm {
namespace runtime {
namespace vulkan {

/*!
 * \brief Create a Vulkan module from SPIRV shaders.
 *
 * \param smap Map from function name to SPIRVShader.
 * \param fmap Map from function name to FunctionInfo.
 * \param source Optional SPIRV text (for inspection).
 *
 * Dispatches through the FFI registry ("ffi.Module.create.vulkan").
 * Each SPIRVShader is serialised to ffi::Bytes before crossing the FFI boundary
 * and rehydrated on the runtime side.
 * Requires libtvm_runtime built with USE_VULKAN=ON to have registered the creator.
 */
inline ffi::Module VulkanModuleCreate(std::unordered_map<std::string, SPIRVShader> smap,
                                      ffi::Map<ffi::String, FunctionInfo> fmap,
                                      std::string source) {
  static const auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.vulkan");
  TVM_FFI_CHECK(fcreate.has_value(), RuntimeError)
      << "ffi.Module.create.vulkan is not registered in runtime. "
      << "Link or load libtvm_runtime built with USE_VULKAN=ON.";
  // Serialise each SPIRVShader to ffi::Bytes for the FFI boundary.
  ffi::Map<ffi::String, ffi::Bytes> shader_bytes;
  for (const auto& kv : smap) {
    std::string buf;
    support::BytesOutStream strm(&buf);
    strm.Write(kv.second.flag);
    strm.Write(kv.second.data);
    shader_bytes.Set(kv.first, ffi::Bytes(std::move(buf)));
  }
  return (*fcreate)(shader_bytes, fmap, ffi::String(source)).cast<ffi::Module>();
}

}  // namespace vulkan

using vulkan::VulkanModuleCreate;
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_MODULE_H_
