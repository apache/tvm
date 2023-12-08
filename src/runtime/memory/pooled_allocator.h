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
 * \file src/runtime/memory/pooled_allocator.h
 */
#ifndef TVM_RUNTIME_MEMORY_POOLED_ALLOCATOR_H_
#define TVM_RUNTIME_MEMORY_POOLED_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace memory {

class PooledAllocator final : public Allocator {
 public:
  static constexpr size_t kDefaultPageSize = 4096;

  explicit PooledAllocator(Device dev, size_t page_size = kDefaultPageSize)
      : Allocator(kPooled), page_size_(page_size), used_memory_(0), device_(dev) {}

  ~PooledAllocator() { ReleaseAll(); }

  Buffer Alloc(size_t nbytes, size_t alignment, DLDataType type_hint) override {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    size_t size = ((nbytes + page_size_ - 1) / page_size_) * page_size_;
    auto&& it = memory_pool_.find(size);
    if (it != memory_pool_.end() && !it->second.empty()) {
      auto&& pool = it->second;
      auto ret = pool.back();
      pool.pop_back();
      return ret;
    }
    Buffer buf;
    buf.device = device_;
    buf.size = size;
    buf.alloc_type = kPooled;
    try {
      buf.data = DeviceAPI::Get(device_)->AllocDataSpace(device_, size, alignment, type_hint);
    } catch (InternalError& err) {
      LOG(WARNING) << "PooledAllocator got InternalError during allocation: " << err.message();
      LOG(WARNING) << "Trying to release all unused memory and reallocate...";
      ReleaseAll();
      buf.data = DeviceAPI::Get(device_)->AllocDataSpace(device_, size, alignment, type_hint);
    }

    used_memory_.fetch_add(size, std::memory_order_relaxed);
    VLOG(1) << "allocate " << size << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  Buffer Alloc(ShapeTuple shape, DLDataType type_hint, const std::string& mem_scope) override {
    if (mem_scope.empty() || mem_scope == "global") {
      return Allocator::Alloc(device_, shape, type_hint, mem_scope);
    }
    LOG(FATAL) << "This alloc should be implemented";
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

 private:
  void ReleaseAll() {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    for (auto const& it : memory_pool_) {
      auto const& pool = it.second;
      for (auto const& buf : pool) {
        DeviceAPI::Get(buf.device)->FreeDataSpace(buf.device, buf.data);
      }
    }
    memory_pool_.clear();
    used_memory_ = 0;
    VLOG(1) << "release all buffers";
  }

 private:
  size_t page_size_;
  std::atomic<size_t> used_memory_;
  std::unordered_map<size_t, std::vector<Buffer>> memory_pool_;
  std::recursive_mutex mu_;
  Device device_;
};

}  // namespace memory
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MEMORY_POOLED_ALLOCATOR_H_
