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
 * \file vulkan_fallback_module.h
 * \brief Codegen-facing Vulkan module factory.
 *
 *   `VulkanModuleCreateWithFallback` is the ONLY entry point codegen uses
 *   to construct a Vulkan `ffi::Module`.  It tries the runtime-registered
 *   factory "ffi.Module.create.vulkan" via the FFI registry; on miss
 *   (USE_VULKAN=OFF build) or when the env var TVM_COMPILE_FORCE_FALLBACK
 *   is truthy, it constructs a `VulkanFallbackModuleNode` directly via
 *   the in-process `VulkanFallbackModuleCreate`.
 *
 *   The fallback exists so that codegen can succeed on a build where the
 *   Vulkan runtime is not linked.  The fallback's saved-bytes are
 *   byte-identical to the real module's saved-bytes for the same payload —
 *   the receiver on a USE_VULKAN=ON box reconstructs a real
 *   `VulkanModuleNode` via "ffi.Module.load_from_bytes.vulkan".  See
 *   src/runtime/vulkan/vulkan_module.cc for the real module + on-disk
 *   format.
 *
 *   Vulkan is multi-shader: each kernel is its own SPIR-V binary (packed
 *   as a `SPIRVShader` — flag + uint32_t data segment), keyed by kernel
 *   name.  The unified per-kernel `smap` payload is `Map<String, Bytes>`
 *   where each value is the serialized `SPIRVShader` bytes.
 */
#ifndef TVM_TARGET_VULKAN_VULKAN_FALLBACK_MODULE_H_
#define TVM_TARGET_VULKAN_VULKAN_FALLBACK_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <utility>

#include "../../runtime/metadata.h"
#include "../../support/env.h"

namespace tvm {
namespace target {

/*!
 * \brief Construct a `VulkanFallbackModuleNode` directly (no FFI registry
 *        round-trip).  Used by `VulkanModuleCreateWithFallback` and by
 *        tests that explicitly want a fallback instance.
 */
ffi::Module VulkanFallbackModuleCreate(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                       ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                       ffi::Map<ffi::String, ffi::String> source);

/*!
 * \brief Codegen-time Vulkan module factory.  Tries the FFI-registered
 *        "ffi.Module.create.vulkan"; on miss or when forced via env var,
 *        falls back to `VulkanFallbackModuleCreate`.
 *
 *   - Registry hit (USE_VULKAN=ON build) → real `VulkanModuleNode` returned.
 *   - Registry miss (USE_VULKAN=OFF build) → `VulkanFallbackModuleNode`
 *     returned.
 *   - `TVM_COMPILE_FORCE_FALLBACK` env var truthy → fallback regardless of
 *     registry state (used by per-backend fallback tests on a USE_X=ON CI
 *     box).
 */
inline ffi::Module VulkanModuleCreateWithFallback(ffi::Map<ffi::String, ffi::Bytes> smap,
                                                  ffi::String fmt,
                                                  ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                                  ffi::Map<ffi::String, ffi::String> source) {
  if (tvm::support::GetEnv<bool>("TVM_COMPILE_FORCE_FALLBACK", false)) {
    return VulkanFallbackModuleCreate(std::move(smap), std::move(fmt), std::move(fmap),
                                      std::move(source));
  }
  // Registry: "ffi.Module.create.vulkan" — real Vulkan runtime factory.
  // Grep hint: grep -rn 'ffi.Module.create.vulkan' src/
  auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.vulkan");
  if (fcreate.has_value()) {
    return (*fcreate)(smap, fmt, fmap, source).cast<ffi::Module>();
  }
  return VulkanFallbackModuleCreate(std::move(smap), std::move(fmt), std::move(fmap),
                                    std::move(source));
}

}  // namespace target
}  // namespace tvm
#endif  // TVM_TARGET_VULKAN_VULKAN_FALLBACK_MODULE_H_
