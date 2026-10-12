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

// Adapt backend-owned tensors to the legacy VM allocator interface.
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <utility>

#include "memory_manager.h"

namespace tvm {
namespace runtime {
namespace memory {

// The VM owns its legacy Buffer/Storage adaptation. Backends expose only owning
// tensors and physical views, so ordinary backend allocations never depend on it.
class OpenCLVMAllocator final : public Allocator {
 public:
  explicit OpenCLVMAllocator(AllocatorType type) : Allocator(type) {}

  Buffer Alloc(Device dev, size_t nbytes, size_t alignment, DLDataType dtype) final {
    alignment = std::max<size_t>(alignment, kAllocAlignment);
    TVM_FFI_CHECK_LE(alignment, std::numeric_limits<uint16_t>::max(), ValueError)
        << "OpenCL allocation alignment cannot be represented";
    TVM_FFI_CHECK_LE(nbytes, static_cast<size_t>(std::numeric_limits<int64_t>::max()), ValueError)
        << "OpenCL allocation size overflows";
    int64_t size = static_cast<int64_t>(nbytes / alignment + (nbytes % alignment != 0));
    return Keep(
        dev, nbytes,
        Allocate({size}, DLDataType{kDLUInt, 8, static_cast<uint16_t>(alignment)}, dev, "global"));
  }

  Buffer Alloc(Device dev, ffi::Shape shape, DLDataType dtype, const std::string& scope) final {
    auto owner = Allocate(shape, dtype, dev, scope);
    size_t nbytes = ffi::GetDataSize(owner);
    return Keep(dev, nbytes, std::move(owner));
  }

  void Free(const Buffer& buffer) final {
    ffi::Tensor owner;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = owners_.find(buffer.data);
      TVM_FFI_ICHECK(it != owners_.end());
      owner = std::move(it->second);
      owners_.erase(it);
      used_memory_ -= buffer.size;
    }
  }

  void* CreateView(const Buffer& buffer, ffi::Shape shape, DLDataType dtype,
                   const std::string& scope) final {
    ffi::Tensor backing;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = owners_.find(buffer.data);
      TVM_FFI_ICHECK(it != owners_.end());
      backing = it->second;
    }
    auto f = ffi::Function::GetGlobalRequired("runtime.opencl.create_view");
    auto view = f(backing, shape, dtype, ffi::String(scope)).cast<ffi::Tensor>();
    void* data = view.data_ptr();
    std::lock_guard<std::mutex> lock(mutex_);
    views_.emplace(data, std::move(view));
    return data;
  }

  void FreeView(Device, void* data) final {
    ffi::Tensor owner;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = views_.find(data);
      TVM_FFI_ICHECK(it != views_.end());
      owner = std::move(it->second);
      views_.erase(it);
    }
  }

  void Clear() final { ffi::Function::GetGlobalRequired("runtime.opencl.clear_pool")(); }
  size_t UsedMemory() const final { return used_memory_.load(std::memory_order_relaxed); }

 private:
  static ffi::Tensor Allocate(ffi::Shape shape, DLDataType dtype, Device dev,
                              const std::string& scope) {
    return ffi::Function::GetGlobalRequired("runtime.opencl.alloc_tensor")(shape, dtype, dev,
                                                                           ffi::String(scope))
        .cast<ffi::Tensor>();
  }

  Buffer Keep(Device dev, size_t nbytes, ffi::Tensor owner) {
    Buffer result{owner.data_ptr(), nbytes, dev, type()};
    std::lock_guard<std::mutex> lock(mutex_);
    owners_.emplace(result.data, std::move(owner));
    used_memory_ += nbytes;
    return result;
  }

  std::atomic<size_t> used_memory_{0};
  std::mutex mutex_;
  std::unordered_multimap<void*, ffi::Tensor> owners_;
  std::unordered_multimap<void*, ffi::Tensor> views_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("DeviceAllocator.opencl", [](Device, int type) -> void* {
    return new OpenCLVMAllocator(static_cast<AllocatorType>(type));
  });
}

}  // namespace memory
}  // namespace runtime
}  // namespace tvm
