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
 * \file workspace_pool.h
 * \brief Workspace pool utility.
 */
#include "workspace_pool.h"

#include <algorithm>
#include <memory>

namespace tvm {
namespace runtime {

// page size.
constexpr size_t kWorkspacePageSize = 4 << 10;

class WorkspacePool::Pool {
 public:
  // constructor
  Pool() {
    // safe guard header on each list.
    Entry e;
    e.data = nullptr;
    e.size = 0;
    free_list_.push_back(e);
    allocated_.push_back(e);
  }
  // allocate from pool
  void* Alloc(Device dev, DeviceAPI* device, Allocator alloc, size_t nbytes) {
    // Allocate align to page.
    nbytes = (nbytes + (kWorkspacePageSize - 1)) / kWorkspacePageSize * kWorkspacePageSize;
    if (nbytes == 0) nbytes = kWorkspacePageSize;
    Entry e;
    auto allocate = [&]() {
      if (alloc) return alloc(device, dev, nbytes);
      // Preserve the workspace alignment when adapting raw allocations to Empty.
      int64_t size = static_cast<int64_t>(nbytes / kTempAllocaAlignment +
                                          (nbytes % kTempAllocaAlignment != 0));
      return device->Empty(ffi::ShapeView(&size, 1), DLDataType{kDLUInt, 8, kTempAllocaAlignment},
                           dev);
    };
    if (free_list_.size() == 2) {
      e = free_list_.back();
      free_list_.pop_back();
      if (e.size < nbytes) {
        // resize the page
        e.owner = ffi::Tensor();
        e.owner = allocate();
        e.data = e.owner.data_ptr();
        e.size = nbytes;
      }
    } else if (free_list_.size() == 1) {
      e.owner = allocate();
      e.data = e.owner.data_ptr();
      e.size = nbytes;
    } else {
      if (free_list_.back().size >= nbytes) {
        // find smallest fit
        auto it = free_list_.end() - 2;
        for (; it->size >= nbytes; --it) {
        }
        e = *(it + 1);
        free_list_.erase(it + 1);
      } else {
        // resize the page
        e = free_list_.back();
        free_list_.pop_back();
        e.owner = ffi::Tensor();
        e.owner = allocate();
        e.data = e.owner.data_ptr();
        e.size = nbytes;
      }
    }
    allocated_.push_back(e);
    return e.data;
  }
  // free resource back to pool
  void Free(void* data) {
    Entry e;
    if (allocated_.back().data == data) {
      // quick path, last allocated.
      e = allocated_.back();
      allocated_.pop_back();
    } else {
      int index = static_cast<int>(allocated_.size()) - 2;
      for (; index > 0 && allocated_[index].data != data; --index) {
      }
      TVM_FFI_ICHECK_GT(index, 0) << "trying to free things that has not been allocated";
      e = allocated_[index];
      allocated_.erase(allocated_.begin() + index);
    }
    // Clear marks outstanding allocations so they cannot enter a replacement session's cache.
    if (e.size == 0) return;
    if (free_list_.back().size < e.size) {
      free_list_.push_back(e);
    } else if (free_list_.size() == 2) {
      free_list_.push_back(free_list_.back());
      free_list_[1] = e;
    } else {
      size_t i = free_list_.size() - 1;
      free_list_.resize(free_list_.size() + 1);
      for (; e.size < free_list_[i].size; --i) {
        free_list_[i + 1] = free_list_[i];
      }
      free_list_[i + 1] = e;
    }
  }
  void Clear() {
    free_list_.resize(1);
    for (auto& entry : allocated_) entry.size = 0;
  }

  bool Contains(void* data) const {
    return std::any_of(allocated_.begin() + 1, allocated_.end(),
                       [data](const Entry& entry) { return entry.data == data; });
  }

 private:
  /*! \brief a single entry in the pool */
  struct Entry {
    ffi::Tensor owner;
    void* data;
    size_t size;
  };
  /*! \brief List of free items, sorted from small to big size */
  std::vector<Entry> free_list_;
  /*! \brief List of allocated items */
  std::vector<Entry> allocated_;
};

WorkspacePool::WorkspacePool(DLDeviceType device_type, DeviceAPI* device, Allocator alloc)
    : device_type_(device_type), device_(device), alloc_(alloc) {}

WorkspacePool::~WorkspacePool() {
  for (auto& streams : array_) {
    for (auto& [stream, pool] : streams) delete pool;
  }
}

void WorkspacePool::Clear() {
  for (auto& streams : array_) {
    for (auto& [stream, pool] : streams) pool->Clear();
  }
}

void* WorkspacePool::AllocWorkspace(Device dev, size_t size) {
  if (static_cast<size_t>(dev.device_id) >= array_.size()) {
    array_.resize(dev.device_id + 1);
  }
  // Keep the historical size-sorted cache separate for each execution stream.
  auto& pool = array_[dev.device_id][device_->GetCurrentStream(dev)];
  if (pool == nullptr) pool = new Pool();
  return pool->Alloc(dev, device_, alloc_, size);
}

void WorkspacePool::FreeWorkspace(Device dev, void* ptr) {
  TVM_FFI_ICHECK(static_cast<size_t>(dev.device_id) < array_.size());
  // A caller may change streams between allocation and release.
  for (auto& [stream, pool] : array_[dev.device_id]) {
    if (pool->Contains(ptr)) {
      pool->Free(ptr);
      return;
    }
  }
  TVM_FFI_THROW(InternalError) << "trying to free things that has not been allocated";
}

}  // namespace runtime
}  // namespace tvm
