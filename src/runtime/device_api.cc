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
 * \file device_api.cc
 * \brief Device specific implementations
 */
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "workspace_pool.h"

namespace tvm {
namespace runtime {

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = TVMDeviceExtType_End;
  // Get API
  static DeviceAPI* Get(const Device& dev) { return Get(dev.device_type); }
  static DeviceAPI* Get(int dev_type, bool allow_missing = false) {
    return Global()->GetAPI(dev_type, allow_missing);
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
  std::mutex mutex_;
  // constructor
  DeviceAPIManager() { std::fill(api_.begin(), api_.end(), nullptr); }
  // Global static variable.
  static DeviceAPIManager* Global() {
    static DeviceAPIManager* inst = new DeviceAPIManager();
    return inst;
  }
  // Get or initialize API.
  DeviceAPI* GetAPI(int type, bool allow_missing) {
    TVM_FFI_CHECK(type >= 0 && type < kMaxDeviceAPI, ValueError) << "Invalid device type " << type;
    if (api_[type] != nullptr) return api_[type];
    std::lock_guard<std::mutex> lock(mutex_);
    if (api_[type] != nullptr) return api_[type];
    api_[type] = GetAPI(DLDeviceType2Str(type), allow_missing);
    return api_[type];
  }

  DeviceAPI* GetAPI(const std::string name, bool allow_missing) {
    std::string factory = "device_api." + name;
    const auto f = tvm::ffi::Function::GetGlobal(factory);
    if (!f.has_value()) {
      TVM_FFI_ICHECK(allow_missing) << "Device API " << name << " is not enabled.";
      return nullptr;
    }
    void* ptr = (*f)().cast<void*>();
    return static_cast<DeviceAPI*>(ptr);
  }
};

DeviceAPI* DeviceAPI::Get(Device dev, bool allow_missing) {
  return DeviceAPIManager::Get(static_cast<int>(dev.device_type), allow_missing);
}

int DeviceAPI::GetDeviceIndex() {
  TVM_FFI_THROW(NotImplementedError) << "Device does not support GetDeviceIndex.";
}

void DeviceAPI::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  // by default, we can always redirect to the flat memory copy operation.
  size_t nbytes = ffi::GetDataSize(*from);
  TVM_FFI_ICHECK_EQ(nbytes, ffi::GetDataSize(*to));

  TVM_FFI_ICHECK(ffi::IsContiguous(*from) && ffi::IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";
  CopyDataFromTo(from->data, from->byte_offset, to->data, to->byte_offset, nbytes, from->device,
                 to->device, from->dtype, stream);
}

void DeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                               size_t num_bytes, Device dev_from, Device dev_to,
                               DLDataType type_hint, TVMStreamHandle stream) {
  TVM_FFI_THROW(InternalError) << "Device does not support CopyDataFromTo.";
}

TVMStreamHandle DeviceAPI::CreateStream(Device dev) { return nullptr; }

void DeviceAPI::FreeStream(Device dev, TVMStreamHandle stream) {}

void DeviceAPI::SetStream(Device dev, TVMStreamHandle stream) {
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIEnvSetStream(dev.device_type, dev.device_id, stream, nullptr));
}

TVMStreamHandle DeviceAPI::GetCurrentStream(Device dev) {
  return TVMFFIEnvGetStream(dev.device_type, dev.device_id);
}

void DeviceAPI::SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.Device_StreamCreate",
           [](DLDevice dev) {
             return reinterpret_cast<int64_t>(DeviceAPIManager::Get(dev)->CreateStream(dev));
           })
      .def("runtime.Device_StreamFree",
           [](DLDevice dev, int64_t stream) {
             DeviceAPIManager::Get(dev)->FreeStream(dev, reinterpret_cast<TVMStreamHandle>(stream));
           })
      .def("runtime.Device_SetStream",
           [](DLDevice dev, int64_t stream) {
             DeviceAPIManager::Get(dev)->SetStream(dev, reinterpret_cast<TVMStreamHandle>(stream));
           })
      .def("runtime.Device_StreamSync",
           [](DLDevice dev, int64_t stream) {
             DeviceAPIManager::Get(dev)->StreamSync(dev, reinterpret_cast<TVMStreamHandle>(stream));
           })
      .def("runtime.Device_StreamSyncFromTo", [](DLDevice dev, int64_t src, int64_t dst) {
        DeviceAPIManager::Get(dev)->SyncStreamFromTo(dev, reinterpret_cast<TVMStreamHandle>(src),
                                                     reinterpret_cast<TVMStreamHandle>(dst));
      });
}

// set device api
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.GetDeviceAttr",
           [](int device_type, int device_id, ffi::String name) {
             DLDevice dev{static_cast<DLDeviceType>(device_type), device_id};
             DeviceAPI* api = DeviceAPI::Get(dev, name == "exist");
             if (api == nullptr) return ffi::Any(false);
             return api->GetAttr(dev, name);
           })
      .def("runtime.TVMSetStream", [](int device_type, int device_id, void* stream) {
        Device dev;
        dev.device_type = static_cast<DLDeviceType>(device_type);
        dev.device_id = device_id;
        DeviceAPIManager::Get(dev)->SetStream(dev, stream);
      });
}
}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

int TVMBackendSetDevice(int device_type, int device_index) {
  TVM_FFI_SAFE_CALL_BEGIN();
  TVM_FFI_CHECK(device_type > 0 && device_type < TVMDeviceExtType_End, ValueError)
      << "Expected a local device type, got " << device_type;
  TVM_FFI_CHECK_GE(device_index, 0, ValueError) << "Expected a nonnegative device index";
  DLDevice dev{static_cast<DLDeviceType>(device_type), device_index};
  DeviceAPI::Get(dev)->SetDevice(dev);
  TVM_FFI_SAFE_CALL_END();
}

int TVMBackendGetDeviceIndex(int device_type, int* out_device_index) {
  TVM_FFI_SAFE_CALL_BEGIN();
  TVM_FFI_CHECK(out_device_index != nullptr, ValueError) << "out_device_index must not be null";
  TVM_FFI_CHECK(device_type > 0 && device_type < TVMDeviceExtType_End, ValueError)
      << "Expected a local device type, got " << device_type;
  DLDevice dev{static_cast<DLDeviceType>(device_type), 0};
  int device_index = DeviceAPI::Get(dev)->GetDeviceIndex();
  *out_device_index = device_index;
  TVM_FFI_SAFE_CALL_END();
}

namespace {
WorkspacePool* BackendWorkspacePool(DLDevice dev) {
  static thread_local std::unordered_map<int, std::unique_ptr<WorkspacePool>> pools;
  auto& pool = pools[static_cast<int>(dev.device_type)];
  if (!pool) pool = std::make_unique<WorkspacePool>(dev.device_type, DeviceAPI::Get(dev));
  return pool.get();
}
}  // namespace

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size, int dtype_code_hint,
                               int dtype_bits_hint) {
  DLDevice dev{static_cast<DLDeviceType>(device_type), device_id};
  TVM_FFI_CHECK_LE(size, std::numeric_limits<size_t>::max(), ValueError)
      << "Workspace size exceeds addressable memory";
  return BackendWorkspacePool(dev)->AllocWorkspace(dev, static_cast<size_t>(size));
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  DLDevice dev{static_cast<DLDeviceType>(device_type), device_id};
  BackendWorkspacePool(dev)->FreeWorkspace(dev, ptr);
  return 0;
}
