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
 * \file src/runtime/memory/lru_cache_allocator.h
 */
#ifndef TVM_RUNTIME_MEMORY_LRU_CACHE_ALLOCATOR_H_
#define TVM_RUNTIME_MEMORY_LRU_CACHE_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <map>
#include <unordered_map>
#include <optional>
#include <deque>
#include <list>
#include <utility>
#include <iterator>
#include <functional>
#include <numeric>

namespace tvm {
namespace runtime {
namespace memory {

using namespace std::placeholders;

class LRUCacheAllocator final : public Allocator {
 public:
  static constexpr size_t kDefaultPageSize = 4096;
  // cache upper bound for free buffers amount.
  // Unfortunately there is no good statistics regards the good upper bound.
  static constexpr size_t kCacheSize = 256;
  explicit LRUCacheAllocator(Device dev, size_t page_size = kDefaultPageSize)
      : Allocator(kLRUCache)
      , page_size_(page_size)
      , device_(dev)
      , lru_cache_(kCacheSize, dev) {
        api_ = DeviceAPI::Get(device_);
      }

  ~LRUCacheAllocator() {
    lru_cache_.remove_all();
  }

  class LRUCache {
  public:

    using object_t = std::pair<size_t, Buffer>;
    using queue_t = std::list<object_t>;
    using pair_t = std::pair<size_t, queue_t::iterator>;
    using storage_t = std::unordered_map<int, queue_t>;
    using lru_t = std::list<pair_t>;

    LRUCache(size_t count, Device dev)
    : capacity_(count)
    {
      ids_.resize(capacity_);
      iters_map_.resize(capacity_);
      std::iota(ids_.begin(), ids_.end(), 1);
      auto api = DeviceAPI::Get(dev);
      deallocator_ = std::bind(&DeviceAPI::FreeDataSpace, api, _1, _2);
    };

    std::optional<Buffer> get(size_t size) {
      auto it = map_.find(size);
      if (it != map_.end()) {
        auto res = std::prev(it->second.end());
        // remove from all queues
        auto q_it = iters_map_[(*res).first];
        ids_.push_back((*res).first);
        queue_.erase(q_it);

        it->second.pop_back();
        if (it->second.empty()) {
          map_.erase(size);
        }
        return (*res).second;
      }
      return {};
    }

    void put(size_t size, const Buffer& val) {
      if (queue_.size() < capacity_) {
        // the latest objects are in the back of the queue
        add_to_storages(size, val);
        return;
      }
      remove_from_storages();
      add_to_storages(size, val);
    }

    size_t remove_oldest(size_t desired_size) {
      size_t removed = 0;
      while((removed < desired_size) && !queue_.empty()) {
        removed += remove_from_storages();
      }
      return removed;
    }

    void remove_all() {
      while(!queue_.empty()) {
        remove_from_storages();
      }
    }

    size_t get_allocated() {
      return queue_.size();
    }

  private:

    void add_to_storages(size_t size, const Buffer& val) {
      auto id = ids_.back();
      ids_.pop_back();
      auto it = map_[size].insert(map_[size].end(), {id, val});
      iters_map_[id] = queue_.insert(queue_.end(), {id, it});
    }

    size_t remove_from_storages() {
      auto oldestUsedIter = queue_.begin();
      queue_.pop_front();
      auto id = (*oldestUsedIter->second).first;
      auto buff = (*oldestUsedIter->second).second;
      auto size = buff.size;
      map_[size].erase(oldestUsedIter->second);
      if (map_[size].empty()) {
        map_.erase(size);
      }

      deallocator_(buff.device, buff.data);
      ids_.push_back(id);
      return size;
    }

    const size_t capacity_ = kCacheSize;
    // existing free objects cache
    storage_t map_;
    // current lru queue
    lru_t queue_;
    // data management structures
    // list of free id-s
    std::vector<size_t> ids_;
    // id-to-list iterator mapping to simplify object removing from the list
    std::vector<lru_t::iterator> iters_map_;
    std::function<void(Device dev, void* ptr)> deallocator_;
  };

  Buffer Alloc(size_t nbytes, size_t alignment, DLDataType type_hint) override {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    size_t size = ((nbytes + page_size_ - 1) / page_size_) * page_size_;
    auto old_buff = lru_cache_.get(size);
    if (old_buff.has_value()) {
      return old_buff.value();
    }
    Buffer buf;
    buf.device = device_;
    buf.size = size;
    buf.alloc_type = kLRUCache;
    try {
      if (nullptr != api_) {
        buf.data = api_->AllocDataSpace(device_, size, alignment, type_hint);
      }
    } catch (InternalError& err) {
      LOG(WARNING) << "LRUCacheAllocator got InternalError during allocation: " << err.message();
      LOG(WARNING) << "Trying to release all unused memory and reallocate...";
      if (nullptr != api_) {
        auto removed = lru_cache_.remove_oldest(size);
        LOG(WARNING) << "Requested " << size << " bytes. Removed " << removed
             << " bytes." << " Still allocated " << lru_cache_.get_allocated() << " buffers.";
        buf.data = api_->AllocDataSpace(device_, size, alignment, type_hint);
      }
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
    lru_cache_.put(buffer.size, buffer);
    VLOG(1) << "reclaim buffer " << buffer.size;
  }

  void Clear() override {
    lru_cache_.remove_all();
  }

  size_t UsedMemory() const override {
    return used_memory_.load(std::memory_order_relaxed);
  }

 private:
  // deallocation of oldest buffers
  void Release(size_t size) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    size_t released = 0;
    if (nullptr != api_) {
      released = lru_cache_.remove_oldest(size);
    }
    VLOG(1) << "released " << released << " bytes.";
  }

 private:
  size_t page_size_;
  std::atomic<uint64_t> used_memory_ = 0;
  std::recursive_mutex mu_;
  Device device_;
  LRUCache lru_cache_;
  DeviceAPI* api_ = nullptr;
};

}  // namespace memory
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MEMORY_LRU_CACHE_ALLOCATOR_H_
