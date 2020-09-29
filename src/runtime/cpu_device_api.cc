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
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <cstdlib>
#include <cstring>

#include "workspace_pool.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace tvm {
namespace runtime {

/*! \brief Function pointer to custom allocator memalign. */
using MemalignFunctionPtr = void* (*)(size_t, size_t);
/*! \brief Function pointer to custom allocator free. */
using FreeFunctionPtr = void (*)(void*);

class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {}
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }
  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    void* ptr;
    // Use custom allocator if it is set.
    if (custom_memalign_fn_ != nullptr) {
      ptr = custom_memalign_fn_(alignment, nbytes);
      if (ptr == nullptr) throw std::bad_alloc();
      return ptr;
    }
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

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    // Use custom allocator free if it is set.
    if (custom_free_fn_ != nullptr) {
      custom_free_fn_(ptr);
      return;
    }
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {}

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;

  static CPUDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CPUDeviceAPI();
    return inst;
  }

  /*! \brief Optional custom memalign function. */
  MemalignFunctionPtr custom_memalign_fn_ = nullptr;
  /*! \brief Optional custom free function. */
  FreeFunctionPtr custom_free_fn_ = nullptr;
};

struct CPUWorkspacePool : public WorkspacePool {
  CPUWorkspacePool() : WorkspacePool(kDLCPU, CPUDeviceAPI::Global()) {}
};

void* CPUDeviceAPI::AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) {
  return dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->AllocWorkspace(ctx, size);
}

void CPUDeviceAPI::FreeWorkspace(TVMContext ctx, void* data) {
  dmlc::ThreadLocalStore<CPUWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

TVM_REGISTER_GLOBAL("device_api.cpu").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CPUDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("runtime.contrib.set_custom_cpu_allocator")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      CPUDeviceAPI* ptr = CPUDeviceAPI::Global();
      ptr->custom_memalign_fn_ = reinterpret_cast<MemalignFunctionPtr>(args[0].operator void*());
      ptr->custom_free_fn_ = reinterpret_cast<FreeFunctionPtr>(args[1].operator void*());
    });

}  // namespace runtime
}  // namespace tvm
