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
 * \file register.cc
 * \brief ROCm compiler backend static registration.
 */
#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include <cctype>
#include <string>

namespace tvm {

namespace backend {
namespace rocm {

std::string ExtractStringWithPrefix(const std::string& str, const std::string& prefix) {
  if (str.find(prefix) != 0) return "";
  std::size_t pos = prefix.length();
  while (pos < str.length() && (std::isdigit(str[pos]) || std::isalpha(str[pos]))) {
    ++pos;
  }
  return str.substr(prefix.length(), pos - prefix.length());
}

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

ffi::Map<ffi::String, ffi::Any> UpdateROCmAttrs(ffi::Map<ffi::String, ffi::Any> target) {
  CheckOrSetAttr(&target, "mtriple", "amdgcn-amd-amdhsa-hcc");
  std::string arch = "gfx900";
  if (target.count("mcpu")) {
    ffi::String mcpu = Downcast<ffi::String>(target.at("mcpu"));
    arch = ExtractStringWithPrefix(mcpu, "gfx");
    TVM_FFI_CHECK(!arch.empty(), ValueError)
        << "ROCm target gets an invalid GFX version: -mcpu=" << mcpu;
  } else {
    ffi::Any val;
    if (const auto f_get_rocm_arch = tvm::ffi::Function::GetGlobal("tvm_callback_rocm_get_arch")) {
      arch = (*f_get_rocm_arch)().cast<std::string>();
    }
    target.Set("mcpu", ffi::String(arch));
  }

  ffi::Any val;
  int version;
  if (!DetectDeviceFlag({kDLROCM, 0}, runtime::kApiVersion, &val)) {
    LOG(WARNING) << "Unable to detect ROCm version, assuming >= 3.5";
    version = 305;
  } else {
    version = val.cast<int>();
  }
  if (version < 305) {
    ffi::Array<ffi::String> mattr;
    if (target.count("mattr")) {
      mattr = Downcast<ffi::Array<ffi::String>>(target.at("mattr"));
    }
    mattr.push_back("-code-object-v3");
    target.Set("mattr", mattr);
  }
  return target;
}

void RegisterTargetKind() {
  namespace refl = tvm::ffi::reflection;

  TVM_REGISTER_TARGET_KIND("rocm", kDLROCM)
      .add_attr_option<ffi::String>("mcpu")
      .add_attr_option<ffi::String>("mtriple")
      .add_attr_option<ffi::Array<ffi::String>>("mattr")
      // TODO(masahi): Support querying from a target device
      // On RDNA cards, thread_warp_size should be 32
      .add_attr_option<int64_t>("max_num_threads", refl::DefaultValue(256))
      .add_attr_option<int64_t>("max_threads_per_block", refl::DefaultValue(256))
      .add_attr_option<int64_t>("max_shared_memory_per_block", refl::DefaultValue(65536))
      .add_attr_option<int64_t>("thread_warp_size", refl::DefaultValue(64))
      .set_default_keys({"rocm", "gpu"})
      .set_target_canonicalizer(UpdateROCmAttrs);
}

}  // namespace rocm
}  // namespace backend

namespace codegen {
#ifdef TVM_LLVM_VERSION
void RegisterAMDGPUCodegen();
namespace llvm {
void RegisterROCMIntrinRules();
}  // namespace llvm
#endif
}  // namespace codegen
}  // namespace tvm

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::backend::rocm::RegisterTargetKind();
#ifdef TVM_LLVM_VERSION
  tvm::codegen::llvm::RegisterROCMIntrinRules();
  tvm::codegen::RegisterAMDGPUCodegen();
#endif
}
