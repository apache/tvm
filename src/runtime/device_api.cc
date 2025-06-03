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
#include <tvm/ffi/container/ndarray.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>

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
  DeviceAPI* rpc_api_{nullptr};
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
    if (type < kRPCSessMask) {
      if (api_[type] != nullptr) return api_[type];
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[type] != nullptr) return api_[type];
      api_[type] = GetAPI(DLDeviceType2Str(type), allow_missing);
      return api_[type];
    } else {
      if (rpc_api_ != nullptr) return rpc_api_;
      std::lock_guard<std::mutex> lock(mutex_);
      if (rpc_api_ != nullptr) return rpc_api_;
      rpc_api_ = GetAPI("rpc", allow_missing);
      return rpc_api_;
    }
  }
  DeviceAPI* GetAPI(const std::string name, bool allow_missing) {
    std::string factory = "device_api." + name;
    const auto f = tvm::ffi::Function::GetGlobal(factory);
    if (!f.has_value()) {
      ICHECK(allow_missing) << "Device API " << name << " is not enabled.";
      return nullptr;
    }
    void* ptr = (*f)().cast<void*>();
    return static_cast<DeviceAPI*>(ptr);
  }
};

DeviceAPI* DeviceAPI::Get(Device dev, bool allow_missing) {
  return DeviceAPIManager::Get(static_cast<int>(dev.device_type), allow_missing);
}

void* DeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return AllocDataSpace(dev, size, kTempAllocaAlignment, type_hint);
}

static size_t GetDataAlignment(const DLDataType dtype) {
  size_t align = (dtype.bits / 8) * dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

size_t DeviceAPI::GetDataSize(const DLTensor& arr, Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value().empty() || mem_scope.value() == "global") {
    size_t size = 1;
    for (int i = 0; i < arr.ndim; ++i) {
      size *= static_cast<size_t>(arr.shape[i]);
    }
    return ffi::GetDataSize(size, arr.dtype);
  }
  LOG(FATAL) << "Device does not support physical mem computation with "
             << "specified memory scope: " << mem_scope.value();
  return 0;
}

void* DeviceAPI::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value() == "" || mem_scope.value() == "global") {
    // by default, we can always redirect to the flat memory allocations
    DLTensor temp;
    temp.data = nullptr;
    temp.device = dev;
    temp.ndim = ndim;
    temp.dtype = dtype;
    temp.shape = const_cast<int64_t*>(shape);
    temp.strides = nullptr;
    temp.byte_offset = 0;
    size_t size = GetDataSize(temp);
    size_t alignment = GetDataAlignment(temp.dtype);
    return AllocDataSpace(dev, size, alignment, dtype);
  }
  LOG(FATAL) << "Device does not support allocate data space with "
             << "specified memory scope: " << mem_scope.value();
  return nullptr;
}

void DeviceAPI::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  // by default, we can always redirect to the flat memory copy operation.
  size_t nbytes = GetDataSize(*from);
  ICHECK_EQ(nbytes, GetDataSize(*to));

  ICHECK(ffi::IsContiguous(*from) && ffi::IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";
  CopyDataFromTo(from->data, from->byte_offset, to->data, to->byte_offset, nbytes, from->device,
                 to->device, from->dtype, stream);
}

void DeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                               size_t num_bytes, Device dev_from, Device dev_to,
                               DLDataType type_hint, TVMStreamHandle stream) {
  LOG(FATAL) << "Device does not support CopyDataFromTo.";
}

void DeviceAPI::FreeWorkspace(Device dev, void* ptr) { FreeDataSpace(dev, ptr); }

TVMStreamHandle DeviceAPI::CreateStream(Device dev) { return nullptr; }

void DeviceAPI::FreeStream(Device dev, TVMStreamHandle stream) {}

TVMStreamHandle DeviceAPI::GetCurrentStream(Device dev) { return nullptr; }

void DeviceAPI::SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
}

TVM_FFI_REGISTER_GLOBAL("runtime.Device_StreamCreate").set_body_typed([](DLDevice dev) {
  return reinterpret_cast<int64_t>(DeviceAPIManager::Get(dev)->CreateStream(dev));
});

TVM_FFI_REGISTER_GLOBAL("runtime.Device_StreamFree")
    .set_body_typed([](DLDevice dev, int64_t stream) {
      DeviceAPIManager::Get(dev)->FreeStream(dev, reinterpret_cast<TVMStreamHandle>(stream));
    });

TVM_FFI_REGISTER_GLOBAL("runtime.Device_SetStream")
    .set_body_typed([](DLDevice dev, int64_t stream) {
      DeviceAPIManager::Get(dev)->SetStream(dev, reinterpret_cast<TVMStreamHandle>(stream));
    });

TVM_FFI_REGISTER_GLOBAL("runtime.Device_StreamSync")
    .set_body_typed([](DLDevice dev, int64_t stream) {
      DeviceAPIManager::Get(dev)->StreamSync(dev, reinterpret_cast<TVMStreamHandle>(stream));
    });

TVM_FFI_REGISTER_GLOBAL("runtime.Device_StreamSyncFromTo")
    .set_body_typed([](DLDevice dev, int64_t src, int64_t dst) {
      DeviceAPIManager::Get(dev)->SyncStreamFromTo(dev, reinterpret_cast<TVMStreamHandle>(src),
                                                   reinterpret_cast<TVMStreamHandle>(dst));
    });

// set device api
TVM_FFI_REGISTER_GLOBAL(tvm::runtime::symbol::tvm_set_device)
    .set_body_packed([](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
      DLDevice dev;
      dev.device_type = static_cast<DLDeviceType>(args[0].cast<int>());
      dev.device_id = args[1].cast<int>();
      DeviceAPIManager::Get(dev)->SetDevice(dev);
    });

// set device api
TVM_FFI_REGISTER_GLOBAL("runtime.GetDeviceAttr")
    .set_body_packed([](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
      DLDevice dev;
      dev.device_type = static_cast<DLDeviceType>(args[0].cast<int>());
      dev.device_id = args[1].cast<int>();

      DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].cast<int>());
      if (kind == kExist) {
        DeviceAPI* api = DeviceAPIManager::Get(dev.device_type, true);
        if (api != nullptr) {
          api->GetAttr(dev, kind, ret);
        } else {
          *ret = 0;
        }
      } else {
        DeviceAPIManager::Get(dev)->GetAttr(dev, kind, ret);
      }
    });

TVM_FFI_REGISTER_GLOBAL("runtime.TVMSetStream")
    .set_body_typed([](int device_type, int device_id, void* stream) {
      Device dev;
      dev.device_type = static_cast<DLDeviceType>(device_type);
      dev.device_id = device_id;
      DeviceAPIManager::Get(dev)->SetStream(dev, stream);
    });
}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

int TVMBackendGetFuncFromEnv(void* mod_node, const char* func_name, TVMFFIObjectHandle* func) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *func = const_cast<tvm::ffi::FunctionObj*>(
      static_cast<ModuleNode*>(mod_node)->GetFuncFromEnv(func_name)->get());
  TVM_FFI_SAFE_CALL_END();
}

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size, int dtype_code_hint,
                               int dtype_bits_hint) {
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;

  DLDataType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  return DeviceAPIManager::Get(dev)->AllocWorkspace(dev, static_cast<size_t>(size), type_hint);
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  DLDevice dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  DeviceAPIManager::Get(dev)->FreeWorkspace(dev, ptr);
  return 0;
}

int TVMBackendRunOnce(void** handle, int (*f)(void*), void* cdata, int nbytes) {
  if (*handle == nullptr) {
    *handle = reinterpret_cast<void*>(1);
    return (*f)(cdata);
  }
  return 0;
}
