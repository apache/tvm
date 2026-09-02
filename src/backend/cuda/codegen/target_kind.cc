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

int CUDAArchNumberFromComputeVersion(const ffi::String& version) {
  std::string value = version;
  size_t separator = value.find('.');
  TVM_FFI_CHECK(separator != std::string::npos && separator > 0 && separator + 2 == value.size(),
                ValueError)
      << "Invalid CUDA compute capability " << version << "; expected <major>.<minor>";

  int major = 0;
  for (size_t i = 0; i < separator; ++i) {
    TVM_FFI_CHECK(value[i] >= '0' && value[i] <= '9', ValueError)
        << "Invalid CUDA compute capability " << version << "; expected <major>.<minor>";
    major = major * 10 + value[i] - '0';
  }
  TVM_FFI_CHECK(value.back() >= '0' && value.back() <= '9', ValueError)
      << "Invalid CUDA compute capability " << version << "; expected <major>.<minor>";
  return major * 10 + value.back() - '0';
}

ffi::String CUDAArchFromComputeVersion(const ffi::String& version) {
  int arch = CUDAArchNumberFromComputeVersion(version);
  ffi::String suffix = arch >= 90 ? "a" : "";
  return ffi::String("sm_") + std::to_string(arch) + suffix;
}

ffi::Map<ffi::String, ffi::Any> UpdateCUDAAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  if (target.count("arch")) {
    ffi::String archStr = target.at("arch").as_or_throw<ffi::String>();
    TVM_FFI_CHECK(StartsWith(archStr, "sm_"), ValueError)
        << "CUDA target gets an invalid CUDA arch: -arch=" << archStr;
  } else {
    ffi::Any version;
    if (!DetectDeviceFlag({kDLCUDA, 0}, runtime::kComputeVersion, &version)) {
      LOG(WARNING) << "Unable to detect CUDA version, default to \"-arch=sm_50\" instead";
      target.Set("arch", ffi::String("sm_50"));
    } else {
      target.Set("arch", CUDAArchFromComputeVersion(version.cast<ffi::String>()));
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
      arch = CUDAArchNumberFromComputeVersion(version.cast<ffi::String>());
    }
    target.Set("mcpu", ffi::String("sm_") + std::to_string(arch));
  }
  return target;
}

void RegisterTargetKinds() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def("target.cuda_arch_from_compute_version", CUDAArchFromComputeVersion);

  TVM_REGISTER_TARGET_KIND("cuda", kDLCUDA)
      .add_attr_option<ffi::String>("mcpu")
      .add_attr_option<ffi::String>("arch")
      .add_attr_option<int64_t>("max_shared_memory_per_block")
      .add_attr_option<int64_t>("max_threads_per_block")
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(32))
      .add_attr_option<int64_t>("registers_per_block")
      .add_attr_option<int64_t>("l2_cache_size_bytes")
      .add_attr_option<int64_t>("max_num_threads",
                                refl::DefaultValue(1024))  // TODO(@zxybazh): deprecate it
      .set_default_keys({"cuda", "gpu"})
      .set_target_canonicalizer(UpdateCUDAAttrs);

  TVM_REGISTER_TARGET_KIND("nvptx", kDLCUDA)
      .add_attr_option<ffi::String>("mcpu")
      .add_attr_option<ffi::String>("mtriple")
      .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(1024))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(32))
      .set_default_keys({"cuda", "gpu"})
      .set_target_canonicalizer(UpdateNVPTXAttrs);
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() { tvm::backend::cuda::RegisterTargetKinds(); }
