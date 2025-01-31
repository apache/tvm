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
 * \file cpu_device_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <cstdlib>
#include <cstring>

#include "workspace_pool.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/sysinfo.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#endif

namespace tvm {
namespace runtime {
class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {}
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }

    switch (kind) {
      case kExist:
        break;
      case kTotalGlobalMemory: {
#if defined(__linux__) || defined(__ANDROID__)
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
          *rv = static_cast<int64_t>(info.totalram) * info.mem_unit;  // Convert to bytes
        } else {
          *rv = -1;
        }
#elif defined(_WIN32)
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        if (GlobalMemoryStatusEx(&statex)) {
          *rv = static_cast<int64_t>(statex.ullTotalPhys);  // Total physical memory in bytes
        } else {
          *rv = -1;
        }
#elif defined(__APPLE__)
        int64_t mem;
        size_t size = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &size, nullptr, 0) == 0) {
          *rv = mem;
        } else {
          *rv = -1;
        }
#else
        *rv = -1;
#endif
      }
      default:
        break;
    }
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#elif defined(__ANDROID__) && __ANDROID_API__ < 17
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    // posix_memalign is available in android ndk since __ANDROID_API__ >= 17
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0) throw std::bad_alloc();
#endif
    return ptr;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;

  bool SupportsDevicePointerArithmeticsOnHost() final { return true; }

  static CPUDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CPUDeviceAPI();
    return inst;
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
  }
};

struct CPUWorkspacePool : public WorkspacePool {
  CPUWorkspacePool() : WorkspacePool(kDLCPU, CPUDeviceAPI::Global()) {}
};

void* CPUDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->AllocWorkspace(dev, size);
}

void CPUDeviceAPI::FreeWorkspace(Device dev, void* data) {
  dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->FreeWorkspace(dev, data);
}

TVM_REGISTER_GLOBAL("device_api.cpu").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CPUDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace tvm
