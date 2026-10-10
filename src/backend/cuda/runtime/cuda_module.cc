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
 * \file cuda_module.cc
 * \brief CUDAModuleNode — runtime-side, plugin-only.  Reachable from C++ only
 *        through the FFI registry keys "ffi.Module.create.cuda" and
 *        "ffi.Module.load_from_bytes.cuda".  No exported header — codegen-side
 *        construction goes through src/target/cuda/cuda_fallback_module.h.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../runtime/metadata.h"
#include "../../../runtime/pack_args.h"
#include "../../../runtime/thread_storage_scope.h"
#include "../../../support/bytes_io.h"
#include "../launch/launch_config.h"
#include "../launch/launch_plan.h"
#include "../module_metadata.h"

namespace tvm {
namespace runtime {

#ifndef CUDA_DRIVER_CALL
#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      TVM_FFI_THROW(CUDAError) << "" #x " failed with error: " << msg;  \
    }                                                                   \
  }
#endif

// Maximum number of GPU supported in CUDAModule (file-local).
static constexpr const int kMaxNumGPUs = 32;

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class CUDAModuleNode : public ffi::ModuleObj {
 public:
  CUDAModuleNode(ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
                 ffi::Map<ffi::String, ffi::String> source)
      : code_(code), fmt_(fmt), fmap_(fmap), source_(source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~CUDAModuleNode() {
    int previous_device = -1;
    cudaError_t get_device_err = cudaGetDevice(&previous_device);
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        cudaError_t set_err = cudaSetDevice(static_cast<int>(i));
        if (set_err != cudaSuccess && set_err != cudaErrorCudartUnloading) {
          continue;
        }
        CUresult result = cuModuleUnload(module_[i]);
        // Ignore errors during cleanup - context may be shutting down
        (void)result;
      }
    }
    if (get_device_err == cudaSuccess) {
      // Preserve the caller's current device after unloading per-device modules.
      (void)cudaSetDevice(previous_device);
    }
  }

  const char* kind() const final { return "cuda"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    return backend::cuda::SaveModule(fmt_, fmap_, code_, source_);
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    // For known compiled formats, return code as string when format matches.
    if (format == fmt_) {
      return ffi::String(code_.data(), code_.size());
    }
    // Look up the source map for an exact match (e.g. "cuda" returns the
    // original C++ source, populated by codegen at construction time).
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    // Empty-format (`mod.get_source()`) — prefer the `cuda` source if present,
    // else fall back to code-as-string when fmt_ is textual.
    if (format.empty()) {
      if (auto it = source_.find("cuda"); it != source_.end()) {
        return (*it).second;
      }
      if (fmt_ == "ptx" || fmt_ == "cuda") {
        return ffi::String(code_.data(), code_.size());
      }
    }
    return ffi::String();
  }

  // get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), code_.data()));
      static auto nvshmem_init_hook = ffi::Function::GetGlobal("runtime.nvshmem.cumodule_init");
      if (nvshmem_init_hook.has_value()) {
        (*nvshmem_init_hook)(static_cast<void*>(module_[device_id]));
      }
    }
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      TVM_FFI_THROW(CUDAError) << "cuModuleGetFunction " << func_name
                               << " failed with error: " << msg;
    }
    return func;
  }

  std::shared_ptr<tvm_cuda_launch::ResourceState> GetResources(int device_id,
                                                               const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = resources_[device_id][func_name];
    if (!state) state = std::make_shared<tvm_cuda_launch::ResourceState>();
    return state;
  }

 private:
  // The binary data (compiled PTX/cubin/fatbin, or raw CUDA source if fmt == "cuda").
  ffi::Bytes code_;
  // The format of code_.
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // Versioned source, resolved configuration and compilation diagnostics.
  ffi::Map<ffi::String, ffi::String> source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<CUmodule, kMaxNumGPUs> module_;
  // All packed wrappers for one CUfunction share monotonically increasing
  // resource limits, including wrappers returned by repeated GetFunction calls.
  std::array<std::unordered_map<std::string, std::shared_ptr<tvm_cuda_launch::ResourceState>>,
             kMaxNumGPUs>
      resources_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class CUDAWrappedFunc {
 public:
  void Init(CUDAModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr, const std::string& func_name,
            const FunctionInfo& info) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    num_kernel_args_ = info->arg_types.size();
    plan_ = tvm_cuda_launch::PackedLaunchPlan(info->cuda_launch_fields, info->launch_param_tags,
                                              info->cuda_kernel_attrs);
    devices_ = std::make_shared<std::array<DeviceState, kMaxNumGPUs>>();
  }

  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
    using namespace tvm_cuda_launch;
    int device_id;
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    TVM_FFI_CHECK_GE(device_id, 0, ValueError);
    TVM_FFI_CHECK_LT(device_id, kMaxNumGPUs, ValueError);
    auto& device = (*devices_)[device_id];
    CUfunction function;
    {
      std::lock_guard<std::mutex> lock(device.mutex);
      if (!device.function) {
        device.resources = m_->GetResources(device_id, func_name_);
        device.function = m_->GetFunc(device_id, func_name_);
      }
      function = device.function;
    }
    LaunchValues values = Normalize(plan_.Extract(args, num_kernel_args_), plan_.requirements);
    PrepareResources<DriverAPI>(function, device_id, &values, device.resources.get());
    CUstream stream = static_cast<CUstream>(TVMFFIEnvGetStream(kDLCUDA, device_id));
    CUlaunchConfig config{};
    std::array<CUlaunchAttribute, kMaxAttributes> attrs{};
    EncodeDriver(values, plan_.requirements, stream, &config, &attrs);
    CUresult result = cuLaunchKernelEx(&config, function, void_args, nullptr);
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
      const char* msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << values.Int(Field::kGridX) << "," << values.Int(Field::kGridY) << ","
         << values.Int(Field::kGridZ) << "), block=(" << values.Int(Field::kBlockX) << ","
         << values.Int(Field::kBlockY) << "," << values.Int(Field::kBlockZ) << ")\n";
      ffi::String cuda = m_->InspectSource("");
      if (cuda.length() != 0) os << "// func_name=" << func_name_ << "\n" << cuda;
      TVM_FFI_THROW(InternalError) << os.str();
    }
  }

 private:
  struct DeviceState {
    std::mutex mutex;
    CUfunction function{nullptr};
    std::shared_ptr<tvm_cuda_launch::ResourceState> resources;
  };
  CUDAModuleNode* m_;
  ffi::ObjectPtr<ffi::Object> sptr_;
  std::string func_name_;
  size_t num_kernel_args_{0};
  tvm_cuda_launch::PackedLaunchPlan plan_;
  std::shared_ptr<std::array<DeviceState, kMaxNumGPUs>> devices_;
};

ffi::Optional<ffi::Function> CUDAModuleNode::GetFunction(const ffi::String& name) {
  if (name == "__tvm_cuda_binary") {
    auto self = ffi::GetRef<ffi::Module>(this);
    return ffi::Function::FromTyped([self, this]() {
      ffi::Array<ffi::String> names;
      for (auto [name, info] : fmap_) names.push_back(name);
      return ffi::Array<ffi::Any>{code_, fmt_, names};
    });
  }

  ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return ffi::Function();
  FunctionInfo info = opt_info.value();
  CUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info);
  return PackFuncVoidAddr(f, info->arg_types, info->arg_extra_tags);
}

// Construct a CUDAModuleNode from in-memory payload.  When fmt == "cuda" the
// code is raw CUDA C++ source — JIT-compile via the Python callback, then
// re-tag with the resulting compiled format ("ptx" / "cubin").
static ffi::Module CUDAModuleCreateImpl(ffi::Bytes code, ffi::String fmt,
                                        ffi::Map<ffi::String, FunctionInfo> fmap,
                                        ffi::Map<ffi::String, ffi::String> source) {
  backend::cuda::CompileSource(&code, &fmt, &source);
  auto n = ffi::make_object<CUDAModuleNode>(code, fmt, fmap, source);
  return ffi::Module(n);
}

static ffi::Module CUDAModuleLoadFromBytes(const ffi::Bytes& bytes) {
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  ffi::Map<ffi::String, ffi::String> source;
  backend::cuda::LoadModule(bytes, &fmt, &fmap, &code, &source);
  return CUDAModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap), std::move(source));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.cuda" — codegen-time CUDA module factory.
  // Used by src/target/cuda/cuda_fallback_module.h:CUDAModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.cuda" — disk loader.  Only this
  // (real) module registers a loader; the fallback module is codegen-time only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.cuda", CUDAModuleLoadFromBytes)
      .def("ffi.Module.create.cuda",
           [](ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              ffi::Map<ffi::String, ffi::String> source) {
             return CUDAModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                                         std::move(source));
           });
}
}  // namespace runtime
}  // namespace tvm
