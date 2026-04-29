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
 * \file build_vulkan.cc
 * \brief Build SPIRV block
 */

#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <utility>

#include "../../runtime/vulkan/spirv_shader.h"
#include "../../support/bytes_io.h"
#include "../build_common.h"
#include "spirv_utils.h"
#include "vulkan_fallback_module.h"

namespace tvm {
namespace codegen {

ffi::Module BuildSPIRV(IRModule mod, Target target) {
  auto [smap, spirv_text] = LowerToSPIRV(mod, target);
  // Serialize each SPIRVShader to ffi::Bytes for the unified per-kernel
  // smap shape.  Each value is a self-packed SPIRVShader (flag + data
  // vector); the Vulkan runtime (USE_VULKAN=ON) deserializes via the
  // inverse helper in src/runtime/vulkan/vulkan_module.cc.
  ffi::Map<ffi::String, ffi::Bytes> shader_bytes;
  for (auto& kv : smap) {
    std::string buf;
    support::BytesOutStream strm(&buf);
    strm.Write(kv.second);
    shader_bytes.Set(kv.first, ffi::Bytes(std::move(buf)));
  }
  // The aggregated SPIR-V text dump is preserved in the in-memory source
  // map keyed by "spv" — only used by InspectSource and never serialized.
  ffi::Map<ffi::String, ffi::String> source;
  source.Set("spv", std::move(spirv_text));
  return target::VulkanModuleCreateWithFallback(std::move(shader_bytes), ffi::String("vulkan"),
                                                ExtractFuncInfo(mod), std::move(source));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.vulkan",
                        [](IRModule mod, Target target) { return BuildSPIRV(mod, target); });
}

}  // namespace codegen
}  // namespace tvm
