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
 * \file metal_module.mm
 * \brief MetalModuleNode — runtime-side, plugin-only.  Reachable from C++
 *        only through the FFI registry keys "ffi.Module.create.metal" and
 *        "ffi.Module.load_from_bytes.metal".  No exported header — codegen-
 *        side construction goes through src/target/metal/metal_fallback_module.h.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/support/io.h>
#include <array>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include "../../support/bytes_io.h"
#include "../file_utils.h"
#include "../metadata.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "metal_common.h"

namespace tvm {
namespace runtime {

/*! \brief Maximum number of GPU supported in MetalModule. */
static constexpr const int kMetalMaxNumDevice = 32;

// Module to support thread-safe multi-GPU execution.
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class MetalModuleNode final : public ffi::ModuleObj {
 public:
  // Unified factory signature shared with the codegen-side fallback in
  // src/target/metal/metal_fallback_module.h.  The per-kernel `smap`
  // payload is Map<String, Bytes> regardless of whether the format is
  // text MSL ("metal") or compiled metallib ("metallib") — text vs binary
  // distinction lives in `fmt`.
  MetalModuleNode(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                  ffi::Map<ffi::String, FunctionInfo> fmap,
                  ffi::Map<ffi::String, ffi::String> source)
      : smap_(std::move(smap)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  const char* kind() const final { return "metal"; }

  /*! \brief Get the property of the runtime module. */
  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    // 3 fields [fmt][fmap][smap].  Source map is in-memory inspection only
    // and is NEVER serialized — matches the cross-backend rule.
    // MetalFallbackModuleNode::SaveToBytes (in
    // src/target/metal/metal_fallback_module.cc) MUST mirror this format
    // byte-for-byte; see one-way comment there.
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(smap_);
    return ffi::Bytes(std::move(result));
  }
  ffi::String InspectSource(const ffi::String& format) const final {
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      if (auto it = source_.find("metal"); it != source_.end()) {
        return (*it).second;
      }
    }
    return ffi::String();
  }

  // get a from primary context in device_id
  id<MTLComputePipelineState> GetPipelineState(size_t device_id, const std::string& func_name) {
    metal::MetalWorkspace* w = metal::MetalWorkspace::Global();
    TVM_FFI_ICHECK_LT(device_id, w->devices.size());
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
    id<MTLLibrary> lib = nil;
    auto kernel = smap_.find(func_name);
    // Directly lookup kernels
    TVM_FFI_ICHECK(kernel != smap_.end());
    const ffi::Bytes& source = (*kernel).second;

    if (fmt_ == "metal") {
      MTLCompileOptions* opts = [MTLCompileOptions alloc];
      opts.languageVersion = MTLLanguageVersion2_3;
      opts.fastMathEnabled = YES;
      // opts = nil;
      // Per-kernel payload is bytes; treat as UTF-8 MSL source.
      std::string source_str(source.data(), source.size());
      lib = [w->devices[device_id]
          newLibraryWithSource:[NSString stringWithUTF8String:source_str.c_str()]
                       options:opts
                         error:&err_msg];
      [opts dealloc];
      if (lib == nil) {
        LOG(FATAL) << "Fail to compile metal source:"
                   << [[err_msg localizedDescription] UTF8String];
      }
      if (err_msg != nil) {
        LOG(INFO) << "Warning: " << [[err_msg localizedDescription] UTF8String];
      }
    } else {
      // Build from library.
      auto q = dispatch_queue_create("q", DISPATCH_QUEUE_SERIAL);
      auto data = dispatch_data_create(source.data(), source.size(), q,
                                       ^{
                                       });
      lib = [w->devices[device_id] newLibraryWithData:data error:&err_msg];
      if (err_msg != nil || lib == nil) {
        LOG(FATAL) << "Fail to compile metal lib:" << [[err_msg localizedDescription] UTF8String];
      }
    }
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:func_name.c_str()]];
    TVM_FFI_ICHECK(f != nil) << "cannot find function " << func_name;
    id<MTLComputePipelineState> state =
        [w->devices[device_id] newComputePipelineStateWithFunction:f error:&err_msg];
    TVM_FFI_ICHECK(state != nil) << "cannot get state:"
                                 << " for function " << func_name
                                 << [[err_msg localizedDescription] UTF8String];
    [f release];
    [lib release];
    // The state.threadExecutionWidth can change dynamically according
    // to the resource constraint in kernel, so it is not strictly hold
    // Turn of warp aware optimziation for now.
    // TVM_FFI_ICHECK_EQ(state.threadExecutionWidth, w->warp_size[device_id]);
    if (e.smap[func_name] != nil) [e.smap[func_name] release];
    e.smap[func_name] = state;
    return state;
  }

 private:
  // device specific entry
  struct DeviceEntry {
    // state cache;
    std::unordered_map<std::string, id<MTLComputePipelineState>> smap;

    ~DeviceEntry() {
      for (auto&& kv : smap) {
        [kv.second release];
      }
    }
  };
  // Per-kernel payload: kernel-name -> bytes (MSL source for fmt="metal" /
  // metallib blob for fmt="metallib").
  ffi::Map<ffi::String, ffi::Bytes> smap_;
  // The format ("metal" source / "metallib" compiled).
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
  // function information.
  std::vector<DeviceEntry> finfo_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class MetalWrappedFunc {
 public:
  // initialize the METAL function.
  void Init(MetalModuleNode* m, ffi::ObjectPtr<ffi::Object> sptr, const std::string& func_name,
            size_t num_buffer_args, size_t num_pack_args,
            const ffi::Array<ffi::String>& launch_param_tags) {
    w_ = metal::MetalWorkspace::Global();
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    num_buffer_args_ = num_buffer_args;
    num_pack_args_ = num_pack_args;
    std::fill(scache_.begin(), scache_.end(), (id<MTLComputePipelineState>)nil);
    launch_param_config_.Init(num_buffer_args + num_pack_args, launch_param_tags);
    metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
    int dev_id = t->device.device_id;
    scache_[dev_id] = m->GetPipelineState(dev_id, func_name);
  }
  // invoke the function with void arguments
  void operator()(ffi::PackedArgs args, ffi::Any* rv, const ArgUnion64* pack_args) const {
    AUTORELEASEPOOL {
      metal::MetalThreadEntry* t = metal::MetalThreadEntry::ThreadLocal();
      int device_id = t->device.device_id;
      // obtain the stream
      auto stream =
          metal::MetalWorkspace::Global()->CastStreamOrGetDefault(t->stream[device_id], device_id);

      // skip launching so the error can be printed during sync
      if (stream->HasErrorHappened()) return;

      if (scache_[device_id] == nil) {
        scache_[device_id] = m_->GetPipelineState(device_id, func_name_);
      }
      ThreadWorkLoad wl = launch_param_config_.Extract(args);
      int blockSize = wl.block_dim(0) * wl.block_dim(1) * wl.block_dim(2);
      auto maxTotalThreadsPerThreadgroup = scache_[device_id].maxTotalThreadsPerThreadgroup;
      TVM_FFI_ICHECK_LE(blockSize, maxTotalThreadsPerThreadgroup);
      // Reuse the pending compute encoder to batch dispatches.
      // The encoder is flushed on sync, copy, or buffer deallocation.
      id<MTLComputeCommandEncoder> encoder = stream->GetPendingComputeEncoder(func_name_);
      [encoder setComputePipelineState:scache_[device_id]];
      for (size_t i = 0; i < num_buffer_args_; ++i) {
        void* buf = args[static_cast<int>(i)].cast<void*>();
        [encoder setBuffer:(id<MTLBuffer>)(buf) offset:0 atIndex:i];
      }
      if (num_pack_args_ != 0) {
        [encoder setBytes:pack_args
                   length:num_pack_args_ * sizeof(ArgUnion64)
                  atIndex:num_buffer_args_];
      }
      // launch
      MTLSize dimGrid = MTLSizeMake(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
      MTLSize dimBlock = MTLSizeMake(wl.block_dim(0), wl.block_dim(1), wl.block_dim(2));
      [encoder dispatchThreadgroups:dimGrid threadsPerThreadgroup:dimBlock];
    };
  }

 private:
  // Reference to global workspace.
  metal::MetalWorkspace* w_;
  // internal module
  MetalModuleNode* m_;
  // the resource holder
  ffi::ObjectPtr<ffi::Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Number of buffer arguments
  size_t num_buffer_args_;
  // number of packed arguments.
  size_t num_pack_args_;
  // Device state cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<id<MTLComputePipelineState>, kMetalMaxNumDevice> scache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

ffi::Optional<ffi::Function> MetalModuleNode::GetFunction(const ffi::String& name) {
  ffi::Function ret;
  AUTORELEASEPOOL {
    ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
    TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
    auto opt_info = fmap_.Get(name);
    if (!opt_info.has_value()) {
      return;
    }
    FunctionInfo info = opt_info.value();
    MetalWrappedFunc f;
    size_t num_buffer_args = NumBufferArgs(info->arg_types);
    f.Init(this, sptr_to_self, name, num_buffer_args, info->arg_types.size() - num_buffer_args,
           info->launch_param_tags);
    ret = PackFuncNonBufferArg(f, info->arg_types);
  };
  return ret;
}

static ffi::Module MetalModuleCreateImpl(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                         ffi::Map<ffi::String, FunctionInfo> fmap,
                                         ffi::Map<ffi::String, ffi::String> source) {
  ffi::ObjectPtr<MetalModuleNode> n;
  AUTORELEASEPOOL {
    n = ffi::make_object<MetalModuleNode>(std::move(smap), std::move(fmt), std::move(fmap),
                                          std::move(source));
  };
  return ffi::Module(n);
}

static ffi::Module MetalModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Map<ffi::String, ffi::Bytes> smap;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&smap);
  // Source map is not serialized — reconstructed empty on load.
  return MetalModuleCreateImpl(std::move(smap), std::move(fmt), std::move(fmap),
                               ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.metal" — codegen-time Metal module factory.
  // Used by src/target/metal/metal_fallback_module.h:MetalModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.metal" — disk loader.  Only this
  // (real) module registers a loader; the fallback is codegen-only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.metal", MetalModuleLoadFromBytes)
      .def("ffi.Module.create.metal",
           [](ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
              ffi::Map<ffi::String, FunctionInfo> fmap, ffi::Map<ffi::String, ffi::String> source) {
             return MetalModuleCreateImpl(std::move(smap), std::move(fmt), std::move(fmap),
                                          std::move(source));
           });
}
}  // namespace runtime
}  // namespace tvm
