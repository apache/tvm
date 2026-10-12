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

// Preserve the Storage result expected by generated VM code while the IPC
// backend owns and pools tensors independently of VM memory management.
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <utility>

#include "memory_manager.h"

namespace tvm {
namespace runtime {
namespace memory {

class CUDAIPCVMAllocator final : public Allocator {
 public:
  CUDAIPCVMAllocator() : Allocator(kPooled) {}

  Storage Allocate(ffi::Shape shape, DLDataType dtype) {
    auto owner = ffi::Function::GetGlobalRequired("runtime.disco.cuda_ipc.empty")(shape, dtype)
                     .cast<ffi::Tensor>();
    Buffer buffer{owner.data_ptr(), ffi::GetDataSize(owner), owner.device(), type()};
    std::lock_guard<std::mutex> lock(mutex_);
    owners_.emplace(buffer.data, std::move(owner));
    return Storage(buffer, this);
  }

  Buffer Alloc(Device, size_t, size_t, DLDataType) final {
    TVM_FFI_THROW(InternalError) << "CUDA IPC storage requires a shape and dtype";
  }

  void Free(const Buffer& buffer) final {
    ffi::Tensor owner;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = owners_.find(buffer.data);
      TVM_FFI_ICHECK(it != owners_.end());
      owner = std::move(it->second);
      owners_.erase(it);
    }
  }

  size_t UsedMemory() const final {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& entry : owners_) total += ffi::GetDataSize(entry.second);
    return total;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_multimap<void*, ffi::Tensor> owners_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("runtime.disco.cuda_ipc.alloc_storage",
                                   [](ffi::Shape shape, DLDataType dtype) {
                                     static auto* allocator = new CUDAIPCVMAllocator();
                                     return allocator->Allocate(shape, dtype);
                                   });
}

}  // namespace memory
}  // namespace runtime
}  // namespace tvm
