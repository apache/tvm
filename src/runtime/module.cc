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
 * \file module.cc
 * \brief TVM module system
 */
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/module.h>

#include <cstring>
#include <unordered_set>

#include "file_utils.h"

namespace tvm {
namespace runtime {

bool RuntimeEnabled(const ffi::String& target_str) {
  std::string target = target_str;
  std::string f_name;
  if (target == "cpu") {
    return true;
  } else if (target == "cuda" || target == "gpu") {
    f_name = "device_api.cuda";
  } else if (target == "cl" || target == "opencl") {
    f_name = "device_api.opencl";
  } else if (target == "mtl" || target == "metal") {
    f_name = "device_api.metal";
  } else if (target == "tflite") {
    f_name = "target.runtime.tflite";
  } else if (target == "vulkan") {
    f_name = "device_api.vulkan";
  } else if (target == "rpc") {
    f_name = "device_api.rpc";
  } else if (target == "hexagon") {
    f_name = "device_api.hexagon";
  } else if (target.length() >= 5 && target.substr(0, 5) == "nvptx") {
    f_name = "device_api.cuda";
  } else if (target.length() >= 4 && target.substr(0, 4) == "rocm") {
    f_name = "device_api.rocm";
  } else if (target.length() >= 4 && target.substr(0, 4) == "llvm") {
    const auto pf = tvm::ffi::Function::GetGlobal("codegen.llvm_target_enabled");
    if (!pf.has_value()) return false;
    return (*pf)(target).cast<bool>();
  } else {
    LOG(FATAL) << "Unknown optional runtime " << target;
  }
  return tvm::ffi::Function::GetGlobal(f_name).has_value();
}

#define TVM_INIT_CONTEXT_FUNC(FuncName) \
  TVM_FFI_CHECK_SAFE_CALL(              \
      TVMFFIEnvModRegisterContextSymbol("__" #FuncName, reinterpret_cast<void*>(FuncName)))

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // Initialize the functions
  TVM_INIT_CONTEXT_FUNC(TVMFFIFunctionCall);
  TVM_INIT_CONTEXT_FUNC(TVMFFIErrorSetRaisedFromCStr);
  TVM_INIT_CONTEXT_FUNC(TVMBackendGetFuncFromEnv);
  TVM_INIT_CONTEXT_FUNC(TVMBackendAllocWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendFreeWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelLaunch);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelBarrier);

  refl::GlobalDef().def("runtime.RuntimeEnabled", RuntimeEnabled);
}

#undef TVM_INIT_CONTEXT_FUNC

}  // namespace runtime
}  // namespace tvm
