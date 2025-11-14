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
 * \file cpu_sample_device_api.cc
 * \brief CPU Sample device API implementation for testing
 */
#include <dmlc/thread_local.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "workspace_pool.h"

namespace tvm {
namespace runtime {

class CPUSampleDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {
    std::cout << "[CPU_SAMPLE] SetDevice called: device_type=" << dev.device_type
              << ", device_id=" << dev.device_id << std::endl;
  }

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    std::cout << "[CPU_SAMPLE] GetAttr called: device_type=" << dev.device_type
              << ", device_id=" << dev.device_id << ", kind=" << kind << std::endl;

    if (kind == kExist) {
      *rv = 1;
      std::cout << "[CPU_SAMPLE] GetAttr kExist: returning 1" << std::endl;
    }

    switch (kind) {
      case kExist:
        break;
      case kMaxThreadsPerBlock:
        *rv = 256;
        std::cout << "[CPU_SAMPLE] GetAttr kMaxThreadsPerBlock: returning 256" << std::endl;
        break;
      case kWarpSize:
        *rv = 1;
        std::cout << "[CPU_SAMPLE] GetAttr kWarpSize: returning 1" << std::endl;
        break;
      default:
        std::cout << "[CPU_SAMPLE] GetAttr: unsupported attribute kind " << kind << std::endl;
        break;
    }
  }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    std::cout << "[CPU_SAMPLE] AllocDataSpace called: nbytes=" << nbytes
              << ", alignment=" << alignment << std::endl;

    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(__ANDROID__) && __ANDROID_API__ < 17
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) throw std::bad_alloc();
#endif
    std::cout << "[CPU_SAMPLE] AllocDataSpace: allocated " << nbytes << " bytes at " << ptr
              << std::endl;
    return ptr;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    std::cout << "[CPU_SAMPLE] FreeDataSpace called: ptr=" << ptr << std::endl;
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    std::cout << "[CPU_SAMPLE] FreeDataSpace: freed memory at " << ptr << std::endl;
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    std::cout << "[CPU_SAMPLE] StreamSync called: device_type=" << dev.device_type
              << ", device_id=" << dev.device_id << ", stream=" << stream << std::endl;
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;

  bool SupportsDevicePointerArithmeticsOnHost() final {
    std::cout << "[CPU_SAMPLE] SupportsDevicePointerArithmeticsOnHost: returning true"
              << std::endl;
    return true;
  }

  static CPUSampleDeviceAPI* Global() {
    static auto* inst = new CPUSampleDeviceAPI();
    return inst;
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    std::cout << "[CPU_SAMPLE] CopyDataFromTo called: size=" << size
              << ", from_offset=" << from_offset << ", to_offset=" << to_offset << std::endl;
    memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
    std::cout << "[CPU_SAMPLE] CopyDataFromTo: copied " << size << " bytes" << std::endl;
  }
};

struct CPUSampleWorkspacePool : public WorkspacePool {
  CPUSampleWorkspacePool() : WorkspacePool(static_cast<DLDeviceType>(20), CPUSampleDeviceAPI::Global()) {
    std::cout << "[CPU_SAMPLE] WorkspacePool created" << std::endl;
  }
};

void* CPUSampleDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  std::cout << "[CPU_SAMPLE] AllocWorkspace called: size=" << size << std::endl;
  void* ptr = dmlc::ThreadLocalStore<CPUSampleWorkspacePool>::Get()->AllocWorkspace(dev, size);
  std::cout << "[CPU_SAMPLE] AllocWorkspace: allocated " << size << " bytes at " << ptr
            << std::endl;
  return ptr;
}

void CPUSampleDeviceAPI::FreeWorkspace(Device dev, void* data) {
  std::cout << "[CPU_SAMPLE] FreeWorkspace called: ptr=" << data << std::endl;
  dmlc::ThreadLocalStore<CPUSampleWorkspacePool>::Get()->FreeWorkspace(dev, data);
  std::cout << "[CPU_SAMPLE] FreeWorkspace: freed memory at " << data << std::endl;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  std::cout << "[CPU_SAMPLE] Registering device_api.cpu_sample" << std::endl;
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.cpu_sample", [](ffi::PackedArgs args, ffi::Any* rv) {
    DeviceAPI* ptr = CPUSampleDeviceAPI::Global();
    *rv = static_cast<void*>(ptr);
  });
}

}  // namespace runtime
}  // namespace tvm
