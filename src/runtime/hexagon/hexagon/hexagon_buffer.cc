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

#include <string>
#include <utility>

#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

static size_t GetDataAlignment(const DLDataType dtype) {
  size_t align = (dtype.bits / 8) * dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

HexagonBuffer::HexagonBuffer(int ndim, const int64_t* shape, DLDataType dtype,
                             Optional<String> scope) {
  ICHECK_LE(ndim, 1) << "Hexagon currently only supports flat allocations "
                     << "and arrays of flat allocations.";

  size_t alignment = GetDataAlignment(dtype);
  // TODO(csullivan): Extend to support arrays of allocations.
  // Move assignment from r-value constructed flat allocation.
  *this = HexagonBuffer(shape[0] * (dtype.bits / 8) * dtype.lanes, alignment, scope);
}

HexagonBuffer::HexagonBuffer(size_t nbytes, size_t alignment, Optional<String> scope) {
  void* ptr = nullptr;
  int ret = posix_memalign(&ptr, alignment, nbytes);
  if (ret != 0) {
    throw std::bad_alloc();
  }
  allocations_.push_back(ptr);
  SetStorageScope(scope);
}

HexagonBuffer::HexagonBuffer(void* data, Optional<String> scope) : managed_{false} {
  SetStorageScope(scope);
  allocations_.push_back(data);
}

HexagonBuffer::~HexagonBuffer() {
  if (managed_) {
    for (auto& ptr : allocations_) {
      free(ptr);
    }
  }
}

HexagonBuffer::HexagonBuffer(HexagonBuffer&& other)
    : allocations_(other.allocations_),
      managed_(other.managed_),
      storage_scope_(other.storage_scope_) {
  other.allocations_.clear();
  other.managed_ = false;
  other.storage_scope_ = StorageScope::kDDR;
}

HexagonBuffer& HexagonBuffer::operator=(HexagonBuffer&& other) {
  std::swap(allocations_, other.allocations_);
  std::swap(managed_, other.managed_);
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
