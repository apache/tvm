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
 * \file target_kind.cc
 * \brief CUDA compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <string>

namespace tvm {

namespace backend {
namespace cuda {

bool DetectDeviceFlag(Device device, runtime::DeviceAttrKind flag, ffi::Any* val) {
  using runtime::DeviceAPI;
  DeviceAPI* api = DeviceAPI::Get(device, true);
  if (api == nullptr) {
    return false;
  }
  api->GetAttr(device, runtime::kExist, val);
  int exists = val->cast<int>();
  if (!exists) {
    return false;
  }
  DeviceAPI::Get(device)->GetAttr(device, flag, val);
  return true;
}

void CheckOrSetAttr(ffi::Map<ffi::String, ffi::Any>* attrs, const ffi::String& name,
                    const ffi::String& value) {
  auto iter = attrs->find(name);
  if (iter == attrs->end()) {
    attrs->Set(name, value);
  } else {
    auto str = (*iter).second.try_cast<ffi::String>();
    TVM_FFI_CHECK(str && str.value() == value, ValueError)
        << "Expects \"" << name << "\" to be \"" << value << "\", but gets: " << (*iter).second;
  }
}

bool StartsWith(const ffi::String& str, const char* prefix) {
  return std::string(str).rfind(prefix, 0) == 0;
}

ffi::Map<ffi::String, ffi::Any> UpdateCUDAAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  if (target.count("arch")) {
    ffi::String archStr = target.at("arch").as_or_throw<ffi::String>();
    TVM_FFI_CHECK(StartsWith(archStr, "sm_"), ValueError)
        << "CUDA target gets an invalid CUDA arch: -arch=" << archStr;
  } else {
    int archInt;
    ffi::Any version;
    if (!DetectDeviceFlag({kDLCUDA, 0}, runtime::kComputeVersion, &version)) {
      LOG(WARNING) << "Unable to detect CUDA version, default to \"-arch=sm_50\" instead";
      archInt = 50;
    } else {
      archInt = std::stod(version.cast<std::string>()) * 10 + 0.1;
    }
    if (archInt >= 90) {
      target.Set("arch", ffi::String("sm_") + std::to_string(archInt) + "a");
    } else {
      target.Set("arch", ffi::String("sm_") + std::to_string(archInt));
    }
  }
  return target;
}

ffi::Map<ffi::String, ffi::Any> UpdateNVPTXAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  CheckOrSetAttr(&target, "mtriple", "nvptx64-nvidia-cuda");
  if (target.count("mcpu")) {
    ffi::String mcpu = target.at("mcpu").as_or_throw<ffi::String>();
    TVM_FFI_CHECK(StartsWith(mcpu, "sm_"), ValueError)
        << "NVPTX target gets an invalid CUDA arch: -mcpu=" << mcpu;
  } else {
    int arch;
    ffi::Any version;
    if (!DetectDeviceFlag({kDLCUDA, 0}, runtime::kComputeVersion, &version)) {
      LOG(WARNING) << "Unable to detect CUDA version, default to \"-mcpu=sm_50\" instead";
      arch = 50;
    } else {
      arch = std::stod(version.cast<std::string>()) * 10 + 0.1;
    }
    target.Set("mcpu", ffi::String("sm_") + std::to_string(arch));
  }
  return target;
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace tvm;
  using namespace tvm::backend::cuda;
  namespace refl = tvm::ffi::reflection;

  TargetKindDef("cuda_host").set_default_device_type(kDLCPU).set_default_keys({"cpu"});

  TargetKindDef("cuda")
      .set_default_device_type(kDLCUDA)
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("arch")
      .def_option<int64_t>("max_shared_memory_per_block")
      .def_option<int64_t>("max_threads_per_block")
      .def_option<int64_t>("thread_warp_size", refl::DefaultValue(32))
      .def_option<int64_t>("registers_per_block")
      .def_option<int64_t>("l2_cache_size_bytes")
      .def_option<int64_t>("max_num_threads",
                           refl::DefaultValue(1024))  // TODO(@zxybazh): deprecate it
      .set_default_keys({"cuda", "gpu"})
      .set_target_canonicalizer(UpdateCUDAAttrs);

  TargetKindDef("nvptx")
      .set_default_device_type(kDLCUDA)
      .def_option<ffi::String>("mcpu")
      .def_option<ffi::String>("mtriple")
      .def_option<int64_t>("max_num_threads", refl::DefaultValue(1024))
      .def_option<int64_t>("thread_warp_size", refl::DefaultValue(32))
      .set_default_keys({"cuda", "gpu"})
      .set_target_canonicalizer(UpdateNVPTXAttrs);
}
