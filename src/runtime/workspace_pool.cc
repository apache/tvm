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
#include <array>
#include <limits>
#include <memory>

namespace tvm {
namespace runtime {

namespace {
struct WorkspacePools {
  std::mutex mutex;
  std::vector<WorkspacePool*> pools;
};
WorkspacePools* AllWorkspacePools() {
  // Kept alive through thread-local pool destruction at process exit.
  static auto* registry = new WorkspacePools();
  return registry;
}
}  // namespace

void ResetWorkspacePool(DLDeviceType device_type) {
  auto* registry = AllWorkspacePools();
  std::lock_guard<std::mutex> lock(registry->mutex);
  for (auto* pool : registry->pools) {
    if (pool->device_type_ == device_type) pool->Reset();
  }
}

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
  void* Alloc(Device dev, DeviceAPI* device, size_t nbytes) {
    TVM_FFI_CHECK_LE(nbytes, std::numeric_limits<size_t>::max() - kWorkspacePageSize, ValueError)
        << "Workspace size overflow";
    // Allocate align to page.
    nbytes = (nbytes + (kWorkspacePageSize - 1)) / kWorkspacePageSize * kWorkspacePageSize;
    if (nbytes == 0) nbytes = kWorkspacePageSize;
    Entry e;
    auto allocate = [&]() {
      TVM_FFI_CHECK_LE(nbytes / kTempAllocaAlignment, std::numeric_limits<int64_t>::max(),
                       ValueError)
          << "Workspace shape overflow";
      int64_t shape = static_cast<int64_t>(nbytes / kTempAllocaAlignment +
                                           (nbytes % kTempAllocaAlignment != 0));
      DLDataType dtype{kDLUInt, 8, kTempAllocaAlignment};
      return device->Empty(ffi::ShapeView(&shape, 1), dtype, dev);
    };
    if (free_list_.size() == 2) {
      e = free_list_.back();
      free_list_.pop_back();
      if (e.size < nbytes) {
        // Release the cached page before growing it.
        e.owner = ffi::Tensor();
        e.data = nullptr;
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
        e.data = nullptr;
        e.owner = allocate();
        e.data = e.owner.data_ptr();
        e.size = nbytes;
      }
    }
    e.generation = generation_;
    allocated_.push_back(e);
    return e.data;
  }
  // free resource back to pool
  void Free(void* data) {
    TVM_FFI_CHECK(data != nullptr, ValueError) << "Cannot free a null workspace";
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
    if (e.generation != generation_) return;
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
  void Reset() {
    // Live allocations keep their original owners and are discarded when freed.
    ++generation_;
    free_list_.clear();
    free_list_.emplace_back();
  }
  // Release all resources
  void Release() {
    free_list_.clear();
    allocated_.clear();
  }

 private:
  /*! \brief a single entry in the pool */
  struct Entry {
    uint64_t generation{0};
    ffi::Tensor owner;
    void* data{nullptr};
    size_t size{0};
  };
  uint64_t generation_{0};
  /*! \brief List of free items, sorted from small to big size */
  std::vector<Entry> free_list_;
  /*! \brief List of allocated items */
  std::vector<Entry> allocated_;
};

WorkspacePool::WorkspacePool(DLDeviceType device_type, DeviceAPI* device)
    : device_type_(device_type), device_(device) {
  auto* registry = AllWorkspacePools();
  std::lock_guard<std::mutex> lock(registry->mutex);
  registry->pools.push_back(this);
}

WorkspacePool::~WorkspacePool() {
  auto* registry = AllWorkspacePools();
  std::lock_guard<std::mutex> lock(registry->mutex);
  auto& pools = registry->pools;
  pools.erase(std::remove(pools.begin(), pools.end(), this), pools.end());
  for (size_t i = 0; i < array_.size(); ++i) {
    if (array_[i] != nullptr) {
      array_[i]->Release();
      delete array_[i];
    }
  }
}

void WorkspacePool::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto* pool : array_) {
    if (pool) pool->Reset();
  }
}

void* WorkspacePool::AllocWorkspace(Device dev, size_t size) {
  TVM_FFI_CHECK_GE(dev.device_id, 0, ValueError) << "Invalid workspace device index";
  std::lock_guard<std::mutex> lock(mutex_);
  if (static_cast<size_t>(dev.device_id) >= array_.size()) {
    array_.resize(dev.device_id + 1, nullptr);
  }
  if (array_[dev.device_id] == nullptr) {
    array_[dev.device_id] = new Pool();
  }
  return array_[dev.device_id]->Alloc(dev, device_, size);
}

void WorkspacePool::FreeWorkspace(Device dev, void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  TVM_FFI_ICHECK(static_cast<size_t>(dev.device_id) < array_.size() &&
                 array_[dev.device_id] != nullptr);
  array_[dev.device_id]->Free(ptr);
}

}  // namespace runtime
}  // namespace tvm
