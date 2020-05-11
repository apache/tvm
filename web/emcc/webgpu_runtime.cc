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

/*
 * \file webgpu_runtime.cc
 * \brief WebGPU runtime based on the TVM JS.
 */

// configurations for the dmlc log.
#define DMLC_LOG_CUSTOMIZE 0
#define DMLC_LOG_STACK_TRACE 0
#define DMLC_LOG_DEBUG 0
#define DMLC_LOG_NODATE 1
#define DMLC_LOG_FATAL_THROW 0

#include <dmlc/thread_local.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../../src/runtime/meta_data.h"
#include "../../src/runtime/vulkan/vulkan_shader.h"
#include "../../src/runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

/*! \brief Thread local workspace */
class WebGPUThreadEntry {
 public:
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  WebGPUThreadEntry();
  // get the threadlocal workspace
  static WebGPUThreadEntry* ThreadLocal();
};

// All the implementations are redirectly to the JS side.
class WebGPUDeviceAPI : public DeviceAPI {
 public:
  WebGPUDeviceAPI() {
    auto* fp = tvm::runtime::Registry::Get("wasm.WebGPUDeviceAPI");
    CHECK(fp != nullptr) << "Cannot find wasm.WebGPUContext in the env";
    auto getter = TypedPackedFunc<PackedFunc(std::string)>(*fp);
    alloc_space_ = getter("deviceAllocDataSpace");
    free_space_ = getter("deviceFreeDataSpace");
    copy_to_gpu_ = getter("deviceCopyToGPU");
    copy_from_gpu_ = getter("deviceCopyFromGPU");
    copy_within_gpu_ = getter("deviceCopyWithinGPU");
  }

  void SetDevice(TVMContext ctx) final {}
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    double ptr_number = alloc_space_(nbytes);
    return reinterpret_cast<void*>(static_cast<int64_t>(ptr_number));
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final { return free_space_(ptr); }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    if (static_cast<int>(ctx_from.device_type) == kDLWebGPU &&
        static_cast<int>(ctx_to.device_type) == kDLWebGPU) {
      CHECK_EQ(ctx_from.device_id, ctx_to.device_id);
      copy_within_gpu_(const_cast<void*>(from), from_offset, to, to_offset, size);
    } else if (static_cast<int>(ctx_from.device_type) == kDLWebGPU &&
               ctx_to.device_type == kDLCPU) {
      void* to_ptr = static_cast<uint8_t*>(to) + to_offset;
      copy_from_gpu_(const_cast<void*>(from), from_offset, to_ptr, size);
    } else if (ctx_from.device_type == kDLCPU &&
               static_cast<int>(ctx_to.device_type) == kDLWebGPU) {
      void* from_ptr = static_cast<uint8_t*>(const_cast<void*>(from)) + from_offset;
      copy_to_gpu_(from_ptr, to, to_offset, size);
    } else {
      LOG(FATAL) << "expect copy from/to WebGPU or between WebGPU";
    }
  }

  TVMStreamHandle CreateStream(TVMContext ctx) final {
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }

  void FreeStream(TVMContext ctx, TVMStreamHandle stream) final {
    LOG(FATAL) << "Not implemented";
    return;
  }

  void SyncStreamFromTo(TVMContext ctx, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    LOG(FATAL) << "Not implemented";
    return;
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final { LOG(FATAL) << "Not implemented"; }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    LOG(FATAL) << "Not implemented";
    return;
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final {
    return WebGPUThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    WebGPUThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<WebGPUDeviceAPI>& Global() {
    static std::shared_ptr<WebGPUDeviceAPI> inst = std::make_shared<WebGPUDeviceAPI>();
    return inst;
  }

 private:
  // NOTE: js return number as double.
  TypedPackedFunc<double(int64_t nbytes)> alloc_space_;
  TypedPackedFunc<void(void* ptr)> free_space_;
  TypedPackedFunc<void(void* from, void* to, int64_t to_offset, int64_t nbytes)> copy_to_gpu_;
  TypedPackedFunc<void(void* from, int64_t from_offset, void* to, int64_t nbytes)> copy_from_gpu_;
  TypedPackedFunc<void(void* from, int64_t from_offset, void* to, int64_t to_offset,
                       int64_t nbytes)>
      copy_within_gpu_;
};

typedef dmlc::ThreadLocalStore<WebGPUThreadEntry> WebGPUThreadStore;

WebGPUThreadEntry::WebGPUThreadEntry()
    : pool(static_cast<DLDeviceType>(kDLWebGPU), WebGPUDeviceAPI::Global()) {}

WebGPUThreadEntry* WebGPUThreadEntry::ThreadLocal() { return WebGPUThreadStore::Get(); }

class WebGPUModuleNode final : public runtime::ModuleNode {
 public:
  explicit WebGPUModuleNode(std::unordered_map<std::string, VulkanShader> smap,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : smap_(smap), fmap_(fmap), source_(source) {
    auto* fp = tvm::runtime::Registry::Get("wasm.WebGPUCreateShader");
    CHECK(fp != nullptr);
    create_shader_ = *fp;
  }

  const char* type_key() const final { return "webgpu"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    auto it = smap_.find(name);
    if (it != smap_.end()) {
      FunctionInfo info = fmap_.at(name);
      info.name = name;
      std::ostringstream os;
      dmlc::JSONWriter writer(&os);
      info.Save(&writer);
      TVMByteArray arr;
      arr.data = reinterpret_cast<char*>(it->second.data.data());
      arr.size = it->second.data.size() * sizeof(it->second.data[0]);
      return create_shader_(os.str(), arr);
    } else {
      return PackedFunc(nullptr);
    }
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    LOG(FATAL) << "Not implemented";
  }

  void SaveToBinary(dmlc::Stream* stream) final { LOG(FATAL) << "Not implemented"; }

  std::string GetSource(const std::string& format) final {
    // can only return source code.
    return source_;
  }

 private:
  // function information table.
  std::unordered_map<std::string, VulkanShader> smap_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The source
  std::string source_;
  // Callback to get the GPU function.
  TypedPackedFunc<PackedFunc(std::string finfo, TVMByteArray shader_data)> create_shader_;
};

Module WebGPUModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::unordered_map<std::string, VulkanShader> smap;
  std::unordered_map<std::string, FunctionInfo> fmap;

  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&smap);
  return Module(make_object<WebGPUModuleNode>(smap, fmap, ""));
}

// for now webgpu is hosted via a vulkan module.
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_vulkan").set_body_typed(WebGPUModuleLoadBinary);

TVM_REGISTER_GLOBAL("device_api.webgpu").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = WebGPUDeviceAPI::Global().get();
  *rv = static_cast<void*>(ptr);
});

}  // namespace runtime
}  // namespace tvm
