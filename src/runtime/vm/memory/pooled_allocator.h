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
 * \file src/runtime/vm/memory/pooled_allocator.h
 */
#ifndef TVM_RUNTIME_MEMORY_POOLED_ALLOCATOR_H_
#define TVM_RUNTIME_MEMORY_POOLED_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include "memory_manager.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace memory {

class PooledAllocator : public Allocator {
 public:
  static constexpr size_t kDefaultPageSize = 4096;

  explicit PooledAllocator(size_t page_size = kDefaultPageSize)
      : Allocator(kPooled), page_size_(page_size), used_memory_(0) {}

  ~PooledAllocator() { ReleaseAll(); }

  Buffer Alloc(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) override {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    size_t size = ((nbytes + page_size_ - 1) / page_size_) * page_size_;
    auto&& it = memory_pool_.find(size);
    if (it != memory_pool_.end() && !it->second.empty()) {
      auto&& pool = it->second;
      bool addressable = DeviceAPI::Get(dev)->SupportsDevicePointerArithmeticsOnHost();
      auto candidate = std::find_if(pool.begin(), pool.end(), [&](const Buffer& buffer) {
        return !addressable ||
               (alignment != 0 && reinterpret_cast<uintptr_t>(buffer.data) % alignment == 0);
      });
      if (candidate != pool.end()) {
        Buffer ret = *candidate;
        pool.erase(candidate);
        return ret;
      }
    }
    Buffer buf;
    buf.device = dev;
    buf.size = size;
    buf.alloc_type = kPooled;
    try {
      buf.data = DeviceAllocDataSpace(dev, size, alignment, type_hint);
    } catch (tvm::ffi::Error& err) {
      LOG(WARNING) << "PooledAllocator got InternalError during allocation: " << err.what();
      LOG(WARNING) << "Trying to release all unused memory and reallocate...";
      ReleaseAll();
      buf.data = DeviceAllocDataSpace(dev, size, alignment, type_hint);
    }

    used_memory_.fetch_add(size, std::memory_order_relaxed);
    VLOG(1) << "allocate " << size << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  Buffer Alloc(Device dev, ffi::Shape shape, DLDataType type_hint,
               const std::string& mem_scope) override {
    if (AllowMemoryScope(mem_scope)) {
      return Allocator::Alloc(dev, shape, type_hint, mem_scope);
    }
    TVM_FFI_THROW(InternalError) << "This alloc should be implemented";
    return {};
  }

  void Free(const Buffer& buffer) override {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (memory_pool_.find(buffer.size) == memory_pool_.end()) {
      memory_pool_.emplace(buffer.size, std::vector<Buffer>{});
    }
    memory_pool_.at(buffer.size).push_back(buffer);
    VLOG(1) << "reclaim buffer " << buffer.size;
  }

  void Clear() override { ReleaseAll(); }

  size_t UsedMemory() const override { return used_memory_.load(std::memory_order_relaxed); }

 protected:
  virtual void* DeviceAllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                     DLDataType type_hint) {
    TVM_FFI_CHECK(alignment && (alignment & (alignment - 1)) == 0, ValueError)
        << "Allocation alignment must be a power of two";
    size_t natural_alignment = std::max(static_cast<size_t>(kAllocAlignment),
                                        static_cast<size_t>(type_hint.bits / 8) * type_hint.lanes);
    DLDataType storage_dtype = type_hint;
    size_t storage_bytes = nbytes;
    bool align_cpu = dev.device_type == kDLCPU && alignment > natural_alignment;
    if (align_cpu) {
      TVM_FFI_CHECK_LE(nbytes, std::numeric_limits<size_t>::max() - (alignment - 1), ValueError)
          << "Aligned allocation size overflow";
      storage_bytes += alignment - 1;
    } else if (alignment > natural_alignment) {
      TVM_FFI_CHECK_LE(alignment, std::numeric_limits<uint16_t>::max(), ValueError)
          << "Backend allocation alignment cannot be represented";
      storage_dtype = DLDataType{kDLUInt, 8, static_cast<uint16_t>(alignment)};
    }
    size_t element_bits = static_cast<size_t>(storage_dtype.bits) * storage_dtype.lanes;
    TVM_FFI_CHECK_GT(element_bits, 0, ValueError) << "Invalid allocation datatype";
    TVM_FFI_CHECK_LE(storage_bytes, (std::numeric_limits<int64_t>::max() - element_bits) / 8,
                     ValueError)
        << "Allocation size overflow";
    int64_t length = static_cast<int64_t>(
        (static_cast<uint64_t>(storage_bytes) * 8 + element_bits - 1) / element_bits);
    ffi::Tensor owner = DeviceAPI::Get(dev)->Empty(ffi::ShapeView(&length, 1), storage_dtype, dev);
    void* data = owner.data_ptr();
    if (align_cpu) {
      size_t space = ffi::GetDataSize(owner);
      TVM_FFI_ICHECK(std::align(alignment, nbytes, data, space) != nullptr);
    }
    owners_.emplace(data, std::move(owner));
    return data;
  }

  virtual void DeviceFreeDataSpace(Device dev, void* ptr) {
    auto it = owners_.find(ptr);
    TVM_FFI_ICHECK(it != owners_.end()) << "Unknown allocation owner";
    owners_.erase(it);
  }

  virtual void ReleaseAll() {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    for (auto const& it : memory_pool_) {
      auto const& pool = it.second;
      for (auto const& buf : pool) {
        DeviceFreeDataSpace(buf.device, buf.data);
      }
    }
    memory_pool_.clear();
    used_memory_ = 0;
    VLOG(1) << "release all buffers";
  }

 protected:
  std::unordered_multimap<void*, ffi::Tensor> owners_;
  size_t page_size_;
  std::atomic<size_t> used_memory_;
  std::unordered_map<size_t, std::vector<Buffer>> memory_pool_;
  std::recursive_mutex mu_;
};

}  // namespace memory
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MEMORY_POOLED_ALLOCATOR_H_
