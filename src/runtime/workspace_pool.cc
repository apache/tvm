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

#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <unordered_map>

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

namespace {
class WorkspaceTensorPool final : public TensorMemoryPool {
 public:
  explicit WorkspaceTensorPool(DeviceAPI* device) : device_(device) {}

 protected:
  ffi::Tensor AllocBacking(size_t nbytes, size_t alignment, DLDevice dev) final {
    alignment = std::max<size_t>(alignment, kTempAllocaAlignment);
    TVM_FFI_CHECK_LE(alignment, std::numeric_limits<uint16_t>::max(), ValueError)
        << "Workspace alignment cannot be represented";
    int64_t length = static_cast<int64_t>(nbytes / alignment + (nbytes % alignment != 0));
    // Empty remains the unpooled resource owner. Encode alignment in its dtype,
    // then expose byte backing without altering or discarding that owner.
    auto owner = device_->Empty(ffi::ShapeView(&length, 1),
                                DLDataType{kDLUInt, 8, static_cast<uint16_t>(alignment)}, dev);
    struct Backing {
      ffi::Tensor owner;
      void AllocData(DLTensor* t) { t->data = owner.data_ptr(); }
      void FreeData(DLTensor*) {}
    };
    int64_t bytes = static_cast<int64_t>(nbytes);
    return ffi::Tensor::FromNDAlloc(Backing{std::move(owner)}, ffi::ShapeView(&bytes, 1),
                                    DLDataType{kDLUInt, 8, 1}, dev);
  }

 private:
  DeviceAPI* device_;
};
}  // namespace

class WorkspacePool::Pool {
 public:
  void* Alloc(Device dev, DeviceAPI* device, size_t nbytes) {
    TVM_FFI_CHECK_LE(nbytes, static_cast<size_t>(std::numeric_limits<int64_t>::max()), ValueError)
        << "Workspace size overflow";
    // Different streams must not reuse an allocation while earlier work is in
    // flight. The enclosing WorkspacePool already separates host threads/devices.
    auto stream = device->GetCurrentStream(dev);
    ffi::Tensor owner;
    int64_t bytes = static_cast<int64_t>(std::max<size_t>(nbytes, 1));
    if (dev.device_type == kDLOpenCL) {
      // OpenCL's Empty already uses the common pool and its in-order queue.
      // Keep only C-ABI pointer ownership here, not a second cache.
      owner = device->Empty(ffi::ShapeView(&bytes, 1), DLDataType{kDLUInt, 8, 1}, dev);
    } else {
      auto& pool = pools_[stream];
      if (!pool) pool = ffi::make_object<WorkspaceTensorPool>(device);
      auto allocate = [&]() {
        return pool->Empty(ffi::ShapeView(&bytes, 1), DLDataType{kDLUInt, 8, 1}, dev);
      };
      try {
        owner = allocate();
      } catch (const ffi::Error&) {
        pool->Clear();
        owner = allocate();
      } catch (const std::bad_alloc&) {
        pool->Clear();
        owner = allocate();
      }
    }
    void* data = owner.data_ptr();
    TVM_FFI_ICHECK(allocated_.emplace(data, std::move(owner)).second)
        << "Workspace allocation returned an already-live pointer";
    return data;
  }

  void Free(void* data) {
    TVM_FFI_CHECK(data != nullptr, ValueError) << "Cannot free a null workspace";
    auto it = allocated_.find(data);
    TVM_FFI_CHECK(it != allocated_.end(), ValueError) << "Unknown workspace allocation";
    allocated_.erase(it);
  }

  void Reset() {
    for (auto& [stream, pool] : pools_) pool->Clear();
    // Live leases keep old pools and resources alive, but cannot recycle into a
    // new backend session after this reset.
    pools_.clear();
  }

  void Release() {
    Reset();
    allocated_.clear();
  }

 private:
  std::unordered_map<TVMStreamHandle, ffi::ObjectPtr<WorkspaceTensorPool>> pools_;
  std::unordered_map<void*, ffi::Tensor> allocated_;
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
