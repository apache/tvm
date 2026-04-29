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
 * \file rocm_module.cc
 * \brief ROCMModuleNode — runtime-side, plugin-only.  Reachable from C++ only
 *        through the FFI registry keys "ffi.Module.create.rocm" and
 *        "ffi.Module.load_from_bytes.hsaco" / "ffi.Module.load_from_bytes.hip".
 *        No exported header — codegen-side construction goes through
 *        src/target/rocm/rocm_fallback_module.h.
 */
#include <hip/hip_runtime_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <vector>

#include "../../support/bytes_io.h"
#include "../metadata.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "rocm_common.h"

namespace tvm {
namespace runtime {

// Maximum number of GPU supported in ROCMModule (file-local).
static constexpr const int kMaxNumGPUs = 32;

// Module to support thread-safe multi-GPU execution.
// hipModule_t is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class ROCMModuleNode : public ffi::ModuleObj {
 public:
  ROCMModuleNode(ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
                 ffi::Map<ffi::String, ffi::String> source)
      : code_(code), fmt_(fmt), fmap_(fmap), source_(source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~ROCMModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        ROCM_CALL(hipSetDevice(static_cast<int>(i)));
        ROCM_DRIVER_CALL(hipModuleUnload(module_[i]));
      }
    }
  }

  const char* kind() const final { return "hip"; }
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }
  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    // Format: [fmt][fmap][code].  Source map is in-memory inspection only and
    // is NEVER serialized — it is lost on save/load round-trip (matches
    // upstream behavior; the receiver rebuilds source from code bytes if
    // possible).  ROCmFallbackModuleNode::SaveToBytes (in
    // src/target/rocm/rocm_fallback_module.cc) MUST mirror this format
    // byte-for-byte; see one-way comment there.
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(result));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) {
      return ffi::String(code_.data(), code_.size());
    }
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty() || format == "llvm") {
      // Backward-compat: legacy returned `hip_source_` (LLVM IR text from the
      // AMDGPU backend) for both empty-format and "llvm".
      if (auto it = source_.find("hip"); it != source_.end()) {
        return (*it).second;
      }
    }
    return ffi::String();
  }

  // get a CUfunction from primary context in device_id
  hipFunction_t GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope

    if (module_[device_id] == nullptr) {
      ROCM_DRIVER_CALL(hipModuleLoadData(&(module_[device_id]), code_.data()));
    }
    hipFunction_t func;
    hipError_t result = hipModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != hipSuccess) {
      TVM_FFI_THROW(ROCMError) << "hipModuleGetFunction " << func_name
                               << " failed with error: " << hipGetErrorString(result);
    }
    return func;
  }
  // get a global var from primary context in device_id
  hipDeviceptr_t GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      ROCM_DRIVER_CALL(hipModuleLoadData(&(module_[device_id]), code_.data()));
    }
    hipDeviceptr_t global = nullptr;
    size_t nbytes = 0;

    ROCM_DRIVER_CALL(hipModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str()));
    TVM_FFI_ICHECK_EQ(nbytes, expect_nbytes);
    return global;
  }

 private:
  // The compiled binary data (hsaco).
  ffi::Bytes code_;
  // The format of code_ (always "hsaco" — ROCm has no source-JIT path).
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<hipModule_t, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class ROCMWrappedFunc {
 public:
  // initialize the ROCM function.
  void Init(ROCMModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const ffi::Array<ffi::String>& launch_param_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(ffi::PackedArgs args, ffi::Any* rv, void* packed_args,
                  size_t packed_nbytes) const {
    int device_id;
    ROCM_CALL(hipGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    hipStream_t strm = static_cast<hipStream_t>(TVMFFIEnvGetStream(kDLROCM, device_id));

    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, packed_args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &packed_nbytes, HIP_LAUNCH_PARAM_END};
    // HIP supports only extra_args.
    ROCM_DRIVER_CALL(hipModuleLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                           wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
                                           wl.block_dim(2), wl.dyn_shmem_size, strm, nullptr,
                                           reinterpret_cast<void**>(&config)));
  }

 private:
  // internal module
  ROCMModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<hipFunction_t, kMaxNumGPUs> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

ffi::Optional<ffi::Function> ROCMModuleNode::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return std::nullopt;
  FunctionInfo info = opt_info.value();
  ROCMWrappedFunc f;
  f.Init(this, sptr_to_self, name, info->arg_types.size(), info->launch_param_tags);
  return PackFuncPackedArgAligned(f, info->arg_types);
}

static ffi::Module ROCMModuleCreateImpl(ffi::Bytes code, ffi::String fmt,
                                        ffi::Map<ffi::String, FunctionInfo> fmap,
                                        ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<ROCMModuleNode>(code, fmt, fmap, source);
  return ffi::Module(n);
}

static ffi::Module ROCMModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  // Source map is not serialized — it is lost on save/load round-trip.
  return ROCMModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                              ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.rocm" — codegen-time ROCm module factory.
  // Used by src/target/rocm/rocm_fallback_module.h:ROCmModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.hsaco" / ".hip" — disk loaders.
  // Only this (real) module registers a loader; the fallback is codegen-only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.hsaco", ROCMModuleLoadFromBytes)
      .def("ffi.Module.load_from_bytes.hip", ROCMModuleLoadFromBytes)
      .def("ffi.Module.create.rocm",
           [](ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              ffi::Map<ffi::String, ffi::String> source) {
             return ROCMModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                                         std::move(source));
           });
}
}  // namespace runtime
}  // namespace tvm
