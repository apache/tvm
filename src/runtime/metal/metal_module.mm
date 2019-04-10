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
 * \file metal_module.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <array>
#include <string>
#include <mutex>
#include "metal_module.h"
#include "metal_common.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class MetalModuleNode final :public runtime::ModuleNode {
 public:
  explicit MetalModuleNode(std::string data,
                           std::string fmt,
                           std::unordered_map<std::string, FunctionInfo> fmap,
                           std::string source)
      : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
  }
  const char* type_key() const final {
    return "metal";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, fmt_)
        << "Can only save to format=" << fmt_;
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }
  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (source_.length() != 0) {
      return source_;
    } else if (fmt_ == "metal") {
      return data_;
    } else {
      return "";
    }
  }
  // get a from primary context in device_id
  id<MTLComputePipelineState> GetPipelineState(
      size_t device_id, const std::string& func_name) {
    metal::MetalWorkspace* w = metal::MetalWorkspace::Global().get();
    CHECK_LT(device_id, w->devices.size());
    // start lock scope.
    std::lock_guard<std::mutex> lock(mutex_);
    if (finfo_.size() <= device_id) {
      finfo_.resize(device_id + 1, DeviceEntry());
    }
    DeviceEntry& e = finfo_[device_id];
    auto it = e.smap.find(func_name);
    if (it != e.smap.end()) return it->second;
    // compile
    NSError* err_msg = nil;
    if (e.lib == nil) {
      if (fmt_ == "metal") {
        MTLCompileOptions *opts = [MTLCompileOptions alloc];
        // Use the Metal 1.2 for now.
        opts.languageVersion = MTLLanguageVersion1_2;
        opts.fastMathEnabled = YES;
        // opts = nil;
        e.lib = [
            w->devices[device_id]
             newLibraryWithSource:[NSString stringWithUTF8String:data_.c_str()]
             options:opts
             error:&err_msg];
        [opts dealloc];
        if (e.lib == nil) {
          LOG(FATAL) << "Fail to compile metal lib:"
                     << [[err_msg localizedDescription] UTF8String];
        }
        if (err_msg != nil) {
          LOG(INFO) << "Warning: "
                    << [[err_msg localizedDescription] UTF8String];
        }
      } else {
        // Build from library.
        auto q = dispatch_queue_create("q", DISPATCH_QUEUE_SERIAL);
        auto data = dispatch_data_create(
            data_.c_str(), data_.length(), q, ^{});
        e.lib = [
            w->devices[device_id]
             newLibraryWithData:data
             error:&err_msg];
        if (err_msg != nil || e.lib == nil) {
          LOG(FATAL) << "Fail to compile metal lib:"
                     << [[err_msg localizedDescription] UTF8String];
        }
      }
      [e.lib retain];
    }
    id<MTLFunction> f = [
        e.lib
         newFunctionWithName:
           [NSString stringWithUTF8String:func_name.c_str()]];
    CHECK(f != nil) << "cannot find function " << func_name;
    id<MTLComputePipelineState> state =
        [w->devices[device_id]
          newComputePipelineStateWithFunction:f
          error:&err_msg];
    CHECK(state != nil)
        << "cannot get state:" << " for function " << func_name
        << [[err_msg localizedDescription] UTF8String];
    // The state.threadExecutionWidth can change dynamically according
    // to the resource constraint in kernel, so it is not strictly hold
    // Turn of warp aware optimziation for now.
    // CHECK_EQ(state.threadExecutionWidth, w->warp_size[device_id]);
    e.smap[func_name] = [state retain];
    return state;
  }

 private:
  // device specific entry
  struct DeviceEntry {
    // library
    id<MTLLibrary> lib = nil;
    // state cache;
    std::unordered_map<std::string, id<MTLComputePipelineState> > smap;

    ~DeviceEntry() {
      if (lib != nil) [lib release];
      for (auto &&kv : smap) {
        [kv.second release];
      }
    }
  };
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The source
  std::string source_;
  // function information.
  std::vector<DeviceEntry> finfo_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class MetalWrappedFunc {
 public:
  // initialize the METAL function.
  void Init(MetalModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            const std::string& func_name,
            size_t num_buffer_args,
            size_t num_pack_args,
            const std::vector<std::string>& thread_axis_tags) {
    w_ = metal::MetalWorkspace::Global().get();
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    num_buffer_args_ = num_buffer_args;
    num_pack_args_ = num_pack_args;
    std::fill(scache_.begin(), scache_.end(), (id<MTLComputePipelineState>)nil);
    thread_axis_cfg_.Init(num_buffer_args + num_pack_args, thread_axis_tags);
    metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
    int dev_id = t->context.device_id;
    scache_[dev_id] = m->GetPipelineState(dev_id, func_name);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  const ArgUnion* pack_args) const {
    metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
    int device_id = t->context.device_id;
    if (scache_[device_id] == nil) {
      scache_[device_id] = m_->GetPipelineState(device_id, func_name_);
    }
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    id<MTLCommandQueue> queue = w_->GetCommandQueue(t->context);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:scache_[device_id]];
    for (size_t i = 0; i < num_buffer_args_; ++i) {
      void* buf = args[static_cast<int>(i)];
      [encoder setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:i];
    }
    if (num_pack_args_ != 0) {
      [encoder setBytes:pack_args
               length:num_pack_args_ * sizeof(ArgUnion)
               atIndex:num_buffer_args_];
    }
    // launch
    MTLSize dimGrid = MTLSizeMake(
        wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
    MTLSize dimBlock = MTLSizeMake(
        wl.block_dim(0), wl.block_dim(1), wl.block_dim(2));
    [encoder dispatchThreadgroups: dimGrid
             threadsPerThreadgroup: dimBlock];
    [encoder endEncoding];
    [cb commit];
  }

 private:
  // Reference to global workspace.
  metal::MetalWorkspace* w_;
  // internal module
  MetalModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // The name of the function.
  std::string func_name_;
  // Number of buffer arguments
  size_t num_buffer_args_;
  // number of packed arguments.
  size_t num_pack_args_;
  // Device state cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<id<MTLComputePipelineState>, kMetalMaxNumDevice> scache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc MetalModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
      << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  MetalWrappedFunc f;
  size_t num_buffer_args = NumBufferArgs(info.arg_types);
  f.Init(this, sptr_to_self, name,
         num_buffer_args, info.arg_types.size() - num_buffer_args,
         info.thread_axis_tags);
  return PackFuncNonBufferArg(f, info.arg_types);
}

Module MetalModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source) {
  metal::MetalWorkspace::Global()->Init();
  std::shared_ptr<MetalModuleNode> n =
      std::make_shared<MetalModuleNode>(data, fmt, fmap, source);
  return Module(n);
}

// Load module from module.
Module MetalModuleLoadFile(const std::string& file_name,
                           const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return MetalModuleCreate(data, fmt, fmap, "");
}

Module MetalModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return MetalModuleCreate(data, fmt, fmap, "");
}

TVM_REGISTER_GLOBAL("module.loadfile_metal")
.set_body_typed(MetalModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadbinary_metal")
.set_body_typed(MetalModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
