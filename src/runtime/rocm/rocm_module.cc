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
 *  Copyright (c) 2017 by Contributors
 * \file rocm_module.cc
 */
#include <tvm/runtime/registry.h>
#include <hip/hip_runtime_api.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <unordered_map>
#include "rocm_module.h"
#include "rocm_common.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// hipModule_t is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class ROCMModuleNode : public runtime::ModuleNode {
 public:
  explicit ROCMModuleNode(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string hip_source,
                          std::string assembly)
    : data_(data), fmt_(fmt), fmap_(fmap), hip_source_(hip_source), assembly_(assembly) {
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

  const char* type_key() const final {
    return "hip";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;


  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    // note: llvm and asm formats are not laodable, so we don't save them
    CHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) { return data_; }
    if (format == "llvm" || format == "") { return hip_source_; }
    if (format == "asm") { return assembly_; }
    return "";
  }

  // get a CUfunction from primary context in device_id
  hipFunction_t GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope

    if (module_[device_id] == nullptr) {
      ROCM_DRIVER_CALL(hipModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    hipFunction_t func;
    hipError_t result = hipModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != hipSuccess) {
      LOG(FATAL)
          << "ROCMError: hipModuleGetFunction " << func_name
          << " failed with error: " << hipGetErrorString(result);
    }
    return func;
  }
  // get a global var from primary context in device_id
  hipDeviceptr_t GetGlobal(int device_id,
                        const std::string& global_name,
                        size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      ROCM_DRIVER_CALL(hipModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    hipDeviceptr_t global = nullptr;
    size_t nbytes = 0;

    hipError_t result = hipSuccess;
    // ROCM doesn't support hipModuleGetGlobal yet.
    // hipError_t result = hipModuleGetGlobal(&global, &nbytes,
    //                                    module_[device_id], global_name.c_str());
    CHECK_EQ(nbytes, expect_nbytes);
    if (result != hipSuccess) {
      LOG(FATAL)
          << "ROCMError: hipModuleGetGlobal " << global_name
          << " failed with error: " << hipGetErrorString(result);
    }
    return global;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The hip source.
  std::string hip_source_;
  // The gcn asm.
  std::string assembly_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<hipModule_t, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class ROCMWrappedFunc {
 public:
  // initialize the ROCM function.
  void Init(ROCMModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            const std::string& func_name,
            size_t num_void_args,
            const std::vector<std::string>& thread_axis_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    thread_axis_cfg_.Init(num_void_args, thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void* packed_args,
                  size_t packed_nbytes) const {
    int device_id;
    ROCM_CALL(hipGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    hipStream_t strm = static_cast<hipStream_t>(ROCMThreadEntry::ThreadLocal()->stream);

    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    void* config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, packed_args,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &packed_nbytes,
      HIP_LAUNCH_PARAM_END
    };
    // HIP supports only extra_args.
    ROCM_DRIVER_CALL(hipModuleLaunchKernel(
        fcache_[device_id],
        wl.grid_dim(0),
        wl.grid_dim(1),
        wl.grid_dim(2),
        wl.block_dim(0),
        wl.block_dim(1),
        wl.block_dim(2),
        0, strm, nullptr,
        reinterpret_cast<void**>(&config)));
  }

 private:
  // internal module
  ROCMModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<hipFunction_t, kMaxNumGPUs> fcache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};


PackedFunc ROCMModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
      << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  ROCMWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.thread_axis_tags);
  return PackFuncPackedArg(f, info.arg_types);
}

Module ROCMModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string hip_source,
    std::string assembly) {
  std::shared_ptr<ROCMModuleNode> n =
    std::make_shared<ROCMModuleNode>(data, fmt, fmap, hip_source, assembly);
  return Module(n);
}

Module ROCMModuleLoadFile(const std::string& file_name,
                          const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return ROCMModuleCreate(data, fmt, fmap, std::string(), std::string());
}

Module ROCMModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return ROCMModuleCreate(data, fmt, fmap, std::string(), std::string());
}


TVM_REGISTER_GLOBAL("module.loadbinary_hsaco")
.set_body_typed(ROCMModuleLoadBinary);


TVM_REGISTER_GLOBAL("module.loadbinary_hip")
.set_body_typed(ROCMModuleLoadBinary);


TVM_REGISTER_GLOBAL("module.loadfile_hsaco")
.set_body_typed(ROCMModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadfile_hip")
.set_body_typed(ROCMModuleLoadFile);
}  // namespace runtime
}  // namespace tvm
