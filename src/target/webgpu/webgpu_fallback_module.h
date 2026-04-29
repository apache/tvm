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
 * \file webgpu_fallback_module.h
 * \brief Codegen-facing WebGPU module factory.
 *
 *   `WebGPUModuleCreateWithFallback` is the ONLY entry point codegen uses
 *   to construct a WebGPU `ffi::Module`.  It tries the runtime-registered
 *   factory "ffi.Module.create.webgpu" via the FFI registry; on miss it
 *   constructs a `WebGPUFallbackModuleNode` directly.  The fallback exists
 *   so that codegen can succeed on a build where the WebGPU runtime is
 *   not linked.  This setup is helpful for cross compilation where we
 *   compile on one env and run on another.
 */
#ifndef TVM_TARGET_WEBGPU_WEBGPU_FALLBACK_MODULE_H_
#define TVM_TARGET_WEBGPU_WEBGPU_FALLBACK_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <utility>

#include "../../runtime/metadata.h"
#include "../../support/env.h"

namespace tvm {
namespace target {

/*!
 * \brief Construct a `WebGPUFallbackModuleNode` directly (no FFI registry
 *        round-trip).  Used by `WebGPUModuleCreateWithFallback` and by
 *        tests that explicitly want a fallback instance.
 */
ffi::Module WebGPUFallbackModuleCreate(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                       ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                       ffi::Map<ffi::String, ffi::String> source);

/*!
 * \brief Codegen-time WebGPU module factory.  Tries the FFI-registered
 *        "ffi.Module.create.webgpu"; on miss (always on the C++ side
 *        today — no native runtime) or when forced via env var, falls
 *        back to `WebGPUFallbackModuleCreate`.
 *
 *   - Registry hit → real WebGPU module (currently never registered C++-
 *     side; wasm runtime registers only the load-from-bytes loader).
 *   - Registry miss → `WebGPUFallbackModuleNode` returned (the canonical
 *     C++ side for WebGPU).
 *   - `TVM_COMPILE_FORCE_FALLBACK` env var truthy → fallback regardless
 *     (test hook, no behavioral change here since fallback is always
 *     selected anyway).
 */
inline ffi::Module WebGPUModuleCreateWithFallback(ffi::Map<ffi::String, ffi::Bytes> smap,
                                                  ffi::String fmt,
                                                  ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                                  ffi::Map<ffi::String, ffi::String> source) {
  if (tvm::support::GetEnv<bool>("TVM_COMPILE_FORCE_FALLBACK", false)) {
    return WebGPUFallbackModuleCreate(std::move(smap), std::move(fmt), std::move(fmap),
                                      std::move(source));
  }
  // Registry: "ffi.Module.create.webgpu" — real WebGPU runtime factory.
  // Grep hint: grep -rn 'ffi.Module.create.webgpu' src/
  auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.webgpu");
  if (fcreate.has_value()) {
    return (*fcreate)(smap, fmt, fmap, source).cast<ffi::Module>();
  }
  return WebGPUFallbackModuleCreate(std::move(smap), std::move(fmt), std::move(fmap),
                                    std::move(source));
}

}  // namespace target
}  // namespace tvm
#endif  // TVM_TARGET_WEBGPU_WEBGPU_FALLBACK_MODULE_H_
