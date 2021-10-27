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

#include "hexagon_buffer.h"

#include <tvm/runtime/module.h>

#include "HAP_compute_res.h"

#include <string>
#include <utility>

#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

struct Allocation {
  Allocation(size_t size, size_t alignment) {}
  virtual ~Allocation() {}
  Allocation(const Allocation&) = delete;
  void* data_{nullptr};
};

template <HexagonBuffer::StorageScope S>
std::unique_ptr<Allocation> Allocator(size_t size, size_t alignment);

struct VTCMAllocation : public Allocation {
  VTCMAllocation(size_t size, size_t alignment) : Allocation(size, alignment), context_id_(0) {
    compute_res_attr_t res_info;
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&res_info));
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param(&res_info, size, 1));
    context_id_ = HAP_compute_res_acquire(&res_info, 10000);
    if (context_id_) {
      data_ = HAP_compute_res_attr_get_vtcm_ptr(&res_info);
      if (!data_) {
        FARF(ERROR, "ERROR: Allocated VTCM ptr is null.");
        HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
        return;
      }
    } else {
      FARF(ERROR, "ERROR: Unable to acquire requeisted resource.");
      return;
    }
    // FARF(ALWAYS, "VTCMAllocation() - Context ID: %u, VTCM ptr: %p", context_id_, data_);
  }
  ~VTCMAllocation() {
    // FARF(ALWAYS, "~VTCMAllocation() - Context ID: %u, VTCM ptr: %p", context_id_, data_);
    if (context_id_ && data_) {
      HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
      data_ = nullptr;
    }
  }
  unsigned int context_id_;
};

static size_t GetDataAlignment(const DLDataType dtype) {
  size_t align = (dtype.bits / 8) * dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

HexagonBuffer::HexagonBuffer(int ndim, const int64_t* shape, DLDataType dtype,
                             Optional<String> scope) {
  ICHECK_LE(ndim, 1) << "Hexagon currently only supports flat allocations "
                     << "and arrays of flat allocations.";
  HEXAGON_PRINT(ALWAYS, "nd allocator");

  size_t alignment = GetDataAlignment(dtype);
  // TODO(csullivan): Extend to support arrays of allocations.
  // Move assignment from r-value constructed flat allocation.
  *this = HexagonBuffer(shape[0] * (dtype.bits / 8) * dtype.lanes, alignment, scope);
}

template <>
std::unique_ptr<Allocation> Allocator<HexagonBuffer::StorageScope::kDDR>(size_t size,
                                                                         size_t alignment) {
  return nullptr;
}

template <>
std::unique_ptr<Allocation> Allocator<HexagonBuffer::StorageScope::kVTCM>(size_t size,
                                                                          size_t alignment) {
  return std::make_unique<VTCMAllocation>(size, alignment);
}

HexagonBuffer::HexagonBuffer(size_t nbytes, size_t alignment, Optional<String> scope) {
  HEXAGON_PRINT(ALWAYS, "nbytes: %u, alignment: %u", nbytes, alignment);
  SetStorageScope(scope);

  std::unique_ptr<Allocation> alloca = nullptr;
  switch (GetStorageScope()) {
    case StorageScope::kDDR:
      alloca = Allocator<StorageScope::kDDR>(nbytes, alignment);
      break;
    case StorageScope::kVTCM:
      alloca = Allocator<StorageScope::kVTCM>(nbytes, alignment);
      break;
  };

  ICHECK(alloca->data_ != nullptr) << "HexagonBuffer allocation failed; scope: "
                                  << static_cast<uint32_t>(GetStorageScope());

  allocations_.push_back(alloca->data_);
  managed_allocations_.push_back(std::move(alloca));
}

HexagonBuffer::HexagonBuffer(void* data, Optional<String> scope) {
  SetStorageScope(scope);
  allocations_.push_back(data);
}

HexagonBuffer::~HexagonBuffer() {
  managed_allocations_.clear();
}

HexagonBuffer::HexagonBuffer(HexagonBuffer&& other)
    : allocations_(other.allocations_),
      managed_allocations_(std::move(other.managed_allocations_)),
      storage_scope_(other.storage_scope_) {
  other.allocations_.clear();
  other.managed_allocations_.clear();
  other.storage_scope_ = StorageScope::kDDR;
}

HexagonBuffer& HexagonBuffer::operator=(HexagonBuffer&& other) {
  std::swap(allocations_, other.allocations_);
  std::swap(managed_allocations_, other.managed_allocations_);
  std::swap(storage_scope_, other.storage_scope_);
  return *this;
}

void* HexagonBuffer::GetPointer() {
  if (!allocations_.size()) {
    return nullptr;
  }
  return (allocations_.size() > 1) ? allocations_.data() : allocations_[0];
}

HexagonBuffer::StorageScope HexagonBuffer::GetStorageScope() const { return storage_scope_; }

void HexagonBuffer::SetStorageScope(Optional<String> scope) {
  if (!scope.defined()) {
    storage_scope_ = StorageScope::kDDR;
  } else {
    if (scope.value() == "global") {
      storage_scope_ = StorageScope::kDDR;
    } else if (scope.value() == "global.vtcm") {
      storage_scope_ = StorageScope::kVTCM;
    } else {
      CHECK(false) << "Encountered unknown HexagonBuffer storage scope: "
                   << std::string(scope.value());
    }
  }
}

HexagonBuffer* IsHexagonBuffer(DLTensor* tensor) {
  if (TVMDeviceExtType(tensor->device.device_type) == kDLHexagon) {
    return static_cast<HexagonBuffer*>(tensor->data);
  }
  return nullptr;
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
