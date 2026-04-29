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
 * \file vulkan_module.cc
 * \brief Plugin-only Vulkan runtime module.  Built only when USE_VULKAN=ON.
 *        No exported header — codegen-side construction goes through
 *        src/target/vulkan/vulkan_fallback_module.h:VulkanModuleCreateWithFallback,
 *        which dispatches to "ffi.Module.create.vulkan" registered below
 *        when this file is linked into the build.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/support/io.h>

#include <string>
#include <utility>

#include "../../support/bytes_io.h"
#include "spirv_shader.h"
#include "vulkan_wrapped_func.h"

namespace tvm {
namespace runtime {
namespace vulkan {

/*!
 * \brief Deserialize a SPIRVShader from ffi::Bytes.
 * Format: flag (uint32_t) followed by data (vector<uint32_t>) — matches
 * the SPIRVShader::Save format in src/runtime/vulkan/spirv_shader.h.
 */
static SPIRVShader DeserializeSPIRVShader(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  SPIRVShader shader;
  TVM_FFI_ICHECK(stream.Read(&shader));
  return shader;
}

static ffi::Module VulkanModuleCreateImpl(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                          ffi::Map<ffi::String, FunctionInfo> fmap,
                                          ffi::Map<ffi::String, ffi::String> source) {
  // Convert Map<String, Bytes> smap → unordered_map<string, SPIRVShader>
  // for the in-memory module.  SaveToBytes will re-emit Map<String, Bytes>
  // form so the bytes shape stays uniform across backends.
  std::unordered_map<std::string, SPIRVShader> internal_smap;
  for (const auto& kv : smap) {
    internal_smap[std::string(kv.first)] = DeserializeSPIRVShader(kv.second);
  }
  auto n = ffi::make_object<VulkanModuleNode>(std::move(internal_smap), std::move(smap),
                                              std::move(fmt), std::move(fmap), std::move(source));
  return ffi::Module(n);
}

static ffi::Module VulkanModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Map<ffi::String, ffi::Bytes> smap;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&smap);
  // Source map is not serialized — reconstructed empty on load.
  return VulkanModuleCreateImpl(std::move(smap), std::move(fmt), std::move(fmap),
                                ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.vulkan" — codegen-time Vulkan module factory.
  // Used by src/target/vulkan/vulkan_fallback_module.h:VulkanModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.vulkan" — disk loader.  Only this
  // (real) module registers a loader; the fallback is codegen-only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.vulkan", VulkanModuleLoadFromBytes)
      .def("ffi.Module.create.vulkan",
           [](ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
              ffi::Map<ffi::String, FunctionInfo> fmap, ffi::Map<ffi::String, ffi::String> source) {
             return VulkanModuleCreateImpl(std::move(smap), std::move(fmt), std::move(fmap),
                                           std::move(source));
           });
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
