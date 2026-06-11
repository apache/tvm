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
 * \file hexagon_fallback_module.h
 * \brief Codegen-facing Hexagon module factory.
 *
 *   `HexagonModuleCreateWithFallback` is the ONLY entry point codegen uses
 *   to construct a Hexagon `ffi::Module`.  It tries the runtime-registered
 *   factory "ffi.Module.create.hexagon" via the FFI registry; on miss it
 *   constructs a `HexagonFallbackModuleNode` directly.  The fallback exists
 *   so that codegen can succeed on a build where the Hexagon runtime is
 *   not linked.  This setup is helpful for cross compilation where we
 *   compile on one env and run on another.
 */
#ifndef TVM_TARGET_HEXAGON_HEXAGON_FALLBACK_MODULE_H_
#define TVM_TARGET_HEXAGON_HEXAGON_FALLBACK_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <utility>

#include "../../runtime/metadata.h"
#include "../../support/env.h"

namespace tvm {
namespace target {

/*! \brief Source-map type for Hexagon — the only backend with Variant. */
using HexagonSourceMap = ffi::Map<ffi::String, ffi::Variant<ffi::String, ffi::Bytes>>;

/*!
 * \brief Construct a `HexagonFallbackModuleNode` directly (no FFI registry
 *        round-trip).  Used by `HexagonModuleCreateWithFallback` and by
 *        tests that explicitly want a fallback instance.
 */
ffi::Module HexagonFallbackModuleCreate(ffi::Bytes code, ffi::String fmt,
                                        ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                        HexagonSourceMap source);

/*!
 * \brief Codegen-time Hexagon module factory.  Tries the FFI-registered
 *        "ffi.Module.create.hexagon"; on miss or when forced via env var,
 *        falls back to `HexagonFallbackModuleCreate`.
 *
 *   - Registry hit (USE_HEXAGON=ON build) → real `HexagonModuleNode`
 *     returned.
 *   - Registry miss (USE_HEXAGON=OFF build) → `HexagonFallbackModuleNode`
 *     returned.
 *   - `TVM_COMPILE_FORCE_FALLBACK` env var truthy → fallback regardless of
 *     registry state (used by per-backend fallback tests on a USE_X=ON CI
 *     box).
 */
inline ffi::Module HexagonModuleCreateWithFallback(
    ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
    HexagonSourceMap source) {
  if (tvm::support::GetEnv<bool>("TVM_COMPILE_FORCE_FALLBACK", false)) {
    return HexagonFallbackModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                                       std::move(source));
  }
  // Registry: "ffi.Module.create.hexagon" — real Hexagon runtime factory.
  // Grep hint: grep -rn 'ffi.Module.create.hexagon' src/
  auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.hexagon");
  if (fcreate.has_value()) {
    return (*fcreate)(code, fmt, fmap, source).cast<ffi::Module>();
  }
  return HexagonFallbackModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                                     std::move(source));
}

}  // namespace target
}  // namespace tvm
#endif  // TVM_TARGET_HEXAGON_HEXAGON_FALLBACK_MODULE_H_
