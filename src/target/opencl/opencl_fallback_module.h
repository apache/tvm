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
 * \file opencl_fallback_module.h
 * \brief Codegen-facing OpenCL module factory.
 *
 *   `OpenCLModuleCreateWithFallback` is the ONLY entry point codegen uses
 *   to construct an OpenCL `ffi::Module`.  It tries the runtime-registered
 *   factory "ffi.Module.create.opencl" via the FFI registry; on miss it
 *   constructs an `OpenCLFallbackModuleNode` directly.  The fallback exists
 *   so that codegen can succeed on a build where the OpenCL runtime is
 *   not linked.
 */
#ifndef TVM_TARGET_OPENCL_OPENCL_FALLBACK_MODULE_H_
#define TVM_TARGET_OPENCL_OPENCL_FALLBACK_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <utility>

#include "../../runtime/metadata.h"
#include "../../support/env.h"

namespace tvm {
namespace target {

/*!
 * \brief Construct an `OpenCLFallbackModuleNode` directly (no FFI
 *        registry round-trip).  Used by `OpenCLModuleCreateWithFallback`
 *        and by tests that explicitly want a fallback instance.
 */
ffi::Module OpenCLFallbackModuleCreate(ffi::Bytes code, ffi::String fmt,
                                       ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                       ffi::Map<ffi::String, ffi::String> source);

/*!
 * \brief Codegen-time OpenCL module factory.  Tries the FFI-registered
 *        "ffi.Module.create.opencl"; on miss or when forced via env
 *        var, falls back to `OpenCLFallbackModuleCreate`.
 *
 *   - Registry hit (USE_OPENCL=ON build) → real OpenCL module returned.
 *   - Registry miss (USE_OPENCL=OFF build) → `OpenCLFallbackModuleNode`
 *     returned.
 *   - `TVM_COMPILE_FORCE_FALLBACK` env var truthy → fallback regardless
 *     of registry state (used by per-backend fallback tests on a
 *     USE_X=ON CI box).
 */
inline ffi::Module OpenCLModuleCreateWithFallback(ffi::Bytes code, ffi::String fmt,
                                                  ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                                  ffi::Map<ffi::String, ffi::String> source) {
  if (tvm::support::GetEnv<bool>("TVM_COMPILE_FORCE_FALLBACK", false)) {
    return OpenCLFallbackModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                                      std::move(source));
  }
  // Registry: "ffi.Module.create.opencl" — real OpenCL runtime factory.
  // Grep hint: grep -rn 'ffi.Module.create.opencl' src/
  auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.opencl");
  if (fcreate.has_value()) {
    return (*fcreate)(code, fmt, fmap, source).cast<ffi::Module>();
  }
  return OpenCLFallbackModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                                    std::move(source));
}

}  // namespace target
}  // namespace tvm
#endif  // TVM_TARGET_OPENCL_OPENCL_FALLBACK_MODULE_H_
