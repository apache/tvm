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
 *  Optional module when Hexagon runtime is switched to off.
 *  When ffi.Module.create.hexagon is not registered, HexagonModuleCreate (the inline
 *  wrapper) raises a clear RuntimeError.  Fall back to a DeviceSourceModule for
 *  compilation-only (source inspection) workflows instead.
 */
#include "../../runtime/hexagon/hexagon_module.h"
#include "../source/codegen_source_base.h"

namespace tvm {
namespace runtime {

// Register a fallback creator so that compiler-side code that calls
// HexagonModuleCreate() when USE_HEXAGON=OFF still gets a usable
// DeviceSourceModule (for source inspection / serialisation) rather than a
// registry-not-found error.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "ffi.Module.create.hexagon",
      [](ffi::String data, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
         ffi::String /*asm_str*/, ffi::String /*obj_str*/, ffi::String /*ir_str*/,
         ffi::String /*bc_str*/) -> ffi::Module {
        LOG(WARNING) << "Hexagon runtime is not enabled, returning a source module...";
        return codegen::DeviceSourceModuleCreate(std::string(data), std::string(fmt), fmap, "hex");
      });
}

}  // namespace runtime
}  // namespace tvm
