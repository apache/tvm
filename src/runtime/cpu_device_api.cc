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
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

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
  ffi::Any GetAttr(Device dev, const ffi::String& name) final {
    if (name == "exist") return dev.device_id >= 0;
    ffi::Any rv;
    if (name == "total_global_memory") {
#if defined(__linux__) || defined(__ANDROID__)
      struct sysinfo info;
      if (sysinfo(&info) == 0) {
        rv = static_cast<int64_t>(info.totalram) * info.mem_unit;  // Convert to bytes
      } else {
        rv = -1;
      }
#elif defined(_WIN32)
      MEMORYSTATUSEX statex;
      statex.dwLength = sizeof(statex);
      if (GlobalMemoryStatusEx(&statex)) {
        rv = static_cast<int64_t>(statex.ullTotalPhys);  // Total physical memory in bytes
      } else {
        rv = -1;
      }
#elif defined(__APPLE__)
      int64_t mem;
      size_t size = sizeof(mem);
      if (sysctlbyname("hw.memsize", &mem, &size, nullptr, 0) == 0) {
        rv = mem;
      } else {
        rv = -1;
      }
#else
      rv = -1;
#endif
    }
    return rv;
  }
  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<ffi::String> mem_scope = std::nullopt) final {
    TVM_FFI_CHECK(
        !mem_scope.has_value() || mem_scope.value().empty() || mem_scope.value() == "global",
        ValueError)
        << "Unsupported memory scope: " << mem_scope.value_or("");
    struct Allocator {
      CPUDeviceAPI* api;
      void AllocData(DLTensor* tensor) {
        size_t alignment =
            std::max(static_cast<size_t>(kAllocAlignment),
                     static_cast<size_t>((tensor->dtype.bits / 8) * tensor->dtype.lanes));
        tensor->data = api->AllocDataSpace(tensor->device, ffi::GetDataSize(*tensor), alignment,
                                           tensor->dtype);
      }
      void FreeData(DLTensor* tensor) { api->FreeDataSpace(tensor->device, tensor->data); }
    };
    return ffi::Tensor::FromNDAlloc(Allocator{this}, shape, dtype, device);
  }

 private:
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) {
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

  void FreeDataSpace(Device dev, void* ptr) {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

 public:
  void StreamSync(Device dev, TVMStreamHandle stream) final {}

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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.cpu", [](ffi::PackedArgs args, ffi::Any* rv) {
    DeviceAPI* ptr = CPUDeviceAPI::Global();
    *rv = static_cast<void*>(ptr);
  });
}
}  // namespace runtime
}  // namespace tvm
