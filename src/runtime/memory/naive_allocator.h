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
 * \file src/runtime/memory/naive_allocator.h
 */
#ifndef TVM_RUNTIME_MEMORY_NAIVE_ALLOCATOR_H_
#define TVM_RUNTIME_MEMORY_NAIVE_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tvm {
namespace runtime {
namespace memory {

class NaiveAllocator final : public Allocator {
 public:
  explicit NaiveAllocator() : Allocator(kNaive), used_memory_(0) {}

  Buffer Alloc(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    Buffer buf;
    buf.device = dev;
    buf.size = nbytes;
    buf.alloc_type = kNaive;
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
    buf.data = data;
    {
      std::lock_guard<std::mutex> lock(mu_);
      owners_.emplace(buf.data, std::move(owner));
    }
    used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
    DLOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  Buffer Alloc(Device dev, ffi::Shape shape, DLDataType type_hint,
               const std::string& mem_scope) final {
    Buffer buf;
    size_t nbytes = 1;
    for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
      nbytes *= static_cast<size_t>(shape[i]);
    }
    nbytes *= (type_hint.bits * type_hint.lanes + 7) / 8;
    buf.device = dev;
    if (AllowMemoryScope(mem_scope)) {
      auto tmp_buf = Allocator::Alloc(dev, shape, type_hint, mem_scope);
      buf.size = tmp_buf.size;
      buf.data = tmp_buf.data;
      buf.alloc_type = kNaive;
      return buf;
    }

    buf.size = nbytes;
    auto owner = DeviceAPI::Get(dev)->Empty(shape, type_hint, dev, ffi::String(mem_scope));
    buf.data = owner.data_ptr();
    {
      std::lock_guard<std::mutex> lock(mu_);
      owners_.emplace(buf.data, std::move(owner));
    }
    used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
    DLOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
    buf.alloc_type = kNaive;
    return buf;
  }

  void Free(const Buffer& buffer) override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = owners_.find(buffer.data);
      TVM_FFI_ICHECK(it != owners_.end()) << "Unknown allocation owner";
      owners_.erase(it);
    }
    used_memory_.fetch_sub(buffer.size, std::memory_order_relaxed);
    DLOG(INFO) << "free " << buffer.size << " B, used memory " << used_memory_ << " B";
  }

  size_t UsedMemory() const override { return used_memory_.load(std::memory_order_relaxed); }

 private:
  std::atomic<size_t> used_memory_;
  std::mutex mu_;
  std::unordered_multimap<void*, ffi::Tensor> owners_;
};

}  // namespace memory
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MEMORY_NAIVE_ALLOCATOR_H_
