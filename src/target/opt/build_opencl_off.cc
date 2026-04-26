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
 *  Optional module when build opencl is switched to off.
 *  Register fallback creators so that compiler-side code (codegen_opencl.cc)
 *  that calls OpenCLModuleCreate() when USE_OPENCL=OFF still gets a usable
 *  DeviceSourceModule for source inspection / serialisation workflows.
 */
#include <tvm/ffi/reflection/registry.h>

#include "../../runtime/metadata.h"
#include "../source/codegen_source_base.h"

namespace tvm {
namespace runtime {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.create.opencl",
           [](ffi::String data, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              ffi::String /*source*/) -> ffi::Module {
             return codegen::DeviceSourceModuleCreate(std::string(data), std::string(fmt), fmap,
                                                      "opencl");
           })
      .def("ffi.Module.create.opencl.spirv",
           [](ffi::Map<ffi::String, ffi::Bytes> /*shader_bytes*/, ffi::String /*spirv_text*/,
              ffi::Map<ffi::String, FunctionInfo> /*fmap*/) -> ffi::Module {
             TVM_FFI_THROW(InternalError)
                 << "OpenCLModuleCreate (SPIRV) is called but OpenCL is not enabled.";
             TVM_FFI_UNREACHABLE();
           });
}

}  // namespace runtime
}  // namespace tvm
