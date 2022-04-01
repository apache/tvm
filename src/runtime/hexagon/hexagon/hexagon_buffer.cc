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

// TODO(csulivan,adstraw,kparzysz-quic) This should be set on a TVM-wide basis.
#if defined(__hexagon__)
#define TVM_LOG_CUSTOMIZE 1
#endif

#include "hexagon_buffer.h"

#include <tvm/runtime/module.h>

#include "hexagon_common.h"

#if defined(__hexagon__)
#include "HAP_compute_res.h"
#endif

#include <algorithm>
#include <string>
#include <utility>

namespace tvm {
namespace runtime {
namespace hexagon {

int hexagon_user_dma_1d_sync(void* dst, void* src, uint32_t length);

struct Allocation {
  Allocation(size_t allocation_nbytes, size_t alignment)
      : allocation_nbytes_(allocation_nbytes), alignment_(alignment) {}
  virtual ~Allocation() {}
  Allocation(const Allocation&) = delete;
  Allocation& operator=(const Allocation&) = delete;
  Allocation(Allocation&&) = delete;
  Allocation& operator=(Allocation&&) = delete;

  void* data_{nullptr};
  size_t allocation_nbytes_;
  size_t alignment_;
};

struct DDRAllocation : public Allocation {
  DDRAllocation(size_t nbytes, size_t alignment) : Allocation(nbytes, alignment) {
#ifdef _WIN32
    data_ = _aligned_malloc(nbytes, alignment);
    CHECK(data_ != nullptr);
#else
    int ret = posix_memalign(&data_, alignment, nbytes);
    CHECK_EQ(ret, 0);
#endif
  }
  ~DDRAllocation() {
#ifdef _WIN32
    _aligned_free(data_);
#else
    free(data_);
#endif
  }
};

#if defined(__hexagon__)
struct VTCMAllocation : public Allocation {
  VTCMAllocation(size_t nbytes, size_t alignment) : Allocation(nbytes, alignment) {
    compute_res_attr_t res_info;
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&res_info));

    // allocate nbytes of vtcm on a single page
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param(&res_info, /*vtcm_size = */ nbytes,
                                                          /*b_single_page = */ 1));
    context_id_ = HAP_compute_res_acquire(&res_info, /*timeout = */ 10000);

    if (context_id_) {
      data_ = HAP_compute_res_attr_get_vtcm_ptr(&res_info);
      if (!data_) {
        HEXAGON_PRINT(ERROR, "ERROR: Allocated VTCM ptr is null.");
        HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
        return;
      }
    } else {
      HEXAGON_PRINT(ERROR, "ERROR: Unable to acquire requeisted resource.");
      return;
    }
    // HEXAGON_PRINT(ALWAYS, "VTCMAllocation() - Context ID: %u, VTCM ptr: %p", context_id_, data_);
  }
  ~VTCMAllocation() {
    // HEXAGON_PRINT(ALWAYS, "~VTCMAllocation() - Context ID: %u, VTCM ptr: %p", context_id_,
    // data_);
    HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
    data_ = nullptr;
  }
  unsigned int context_id_{0};
};
#else
struct VTCMAllocation : public DDRAllocation {
  VTCMAllocation(size_t nbytes, size_t alignment) : DDRAllocation(nbytes, alignment) {}
};
#endif

template <HexagonBuffer::StorageScope S>
std::unique_ptr<Allocation> Allocator(size_t nbytes, size_t alignment);

template <>
std::unique_ptr<Allocation> Allocator<HexagonBuffer::StorageScope::kDDR>(size_t nbytes,
                                                                         size_t alignment) {
  return std::make_unique<DDRAllocation>(nbytes, alignment);
}

template <>
std::unique_ptr<Allocation> Allocator<HexagonBuffer::StorageScope::kVTCM>(size_t nbytes,
                                                                          size_t alignment) {
  return std::make_unique<VTCMAllocation>(nbytes, alignment);
}

HexagonBuffer::HexagonBuffer(size_t nbytes, size_t alignment, Optional<String> scope)
    : ndim_(1), nbytes_per_allocation_(nbytes) {
  SetStorageScope(scope);

  std::unique_ptr<Allocation> alloca = nullptr;
  if (GetStorageScope() == StorageScope::kDDR) {
    alloca = Allocator<StorageScope::kDDR>(nbytes, alignment);
  } else if (GetStorageScope() == StorageScope::kVTCM) {
    alloca = Allocator<StorageScope::kVTCM>(nbytes, alignment);
  }
  CHECK(alloca != nullptr);
  allocations_.push_back(alloca->data_);
  managed_allocations_.push_back(std::move(alloca));
}

HexagonBuffer::HexagonBuffer(size_t nallocs, size_t nbytes, size_t alignment,
                             Optional<String> scope)
    : ndim_(2), nbytes_per_allocation_(nbytes) {
  SetStorageScope(scope);
  for (size_t i = 0; i < nallocs; ++i) {
    std::unique_ptr<Allocation> alloca = nullptr;
    if (GetStorageScope() == StorageScope::kDDR) {
      alloca = Allocator<StorageScope::kDDR>(nbytes, alignment);
    } else if (GetStorageScope() == StorageScope::kVTCM) {
      alloca = Allocator<StorageScope::kVTCM>(nbytes, alignment);
    }
    CHECK(alloca != nullptr);
    allocations_.push_back(alloca->data_);
    managed_allocations_.push_back(std::move(alloca));
  }
}

HexagonBuffer::HexagonBuffer(void* data, size_t nbytes, Optional<String> scope)
    : ndim_(1), nbytes_per_allocation_(nbytes) {
  SetStorageScope(scope);
  // disallow external VTCM allocations
  CHECK(GetStorageScope() != HexagonBuffer::StorageScope::kVTCM);
  allocations_.push_back(data);
}

HexagonBuffer::~HexagonBuffer() { managed_allocations_.clear(); }

void* HexagonBuffer::GetPointer() {
  ICHECK(allocations_.size())
      << "Internal failure, allocations_ should be set in HexagonBuffer constructor";

  if (ndim_ == 1) {
    ICHECK_EQ(allocations_.size(), 1);
    return allocations_[0];
  } else if (ndim_ == 2) {
    return allocations_.data();
  } else {
    LOG(FATAL) << "HexagonBuffer should be either 1-d or 2-d, not " << ndim_ << "-d";
    return nullptr;
  }
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

void HexagonBuffer::CopyTo(void* data, size_t nbytes) const {
  CHECK_LE(nbytes, TotalBytes());
  CHECK(managed_allocations_.size() && "CopyTo not supported on unmanaged `external` allocations");

  size_t copied = 0;
  for (const auto& managed_alloc : managed_allocations_) {
    size_t bytes_to_copy = std::min(nbytes - copied, managed_alloc->allocation_nbytes_);
    if (bytes_to_copy == 0) break;

    void* data_plus_copied = static_cast<void*>((static_cast<char*>(data) + copied));
    int status = hexagon_user_dma_1d_sync(data_plus_copied, managed_alloc->data_, bytes_to_copy);
    CHECK_EQ(status, 0);

    copied += bytes_to_copy;
  }
}

void HexagonBuffer::CopyFrom(void* data, size_t nbytes) {
  CHECK_LE(nbytes, TotalBytes());
  CHECK(managed_allocations_.size() &&
        "CopyFrom not supported on unmanaged `external` allocations");

  size_t copied = 0;
  for (const auto& managed_alloc : managed_allocations_) {
    size_t bytes_to_copy = std::min(nbytes - copied, managed_alloc->allocation_nbytes_);
    if (bytes_to_copy == 0) break;

    void* data_plus_copied = static_cast<void*>((static_cast<char*>(data) + copied));
    int status = hexagon_user_dma_1d_sync(managed_alloc->data_, data_plus_copied, bytes_to_copy);
    CHECK_EQ(status, 0);

    copied += bytes_to_copy;
  }
}

void HexagonBuffer::CopyFrom(const HexagonBuffer& other, size_t nbytes) {
  CHECK_LE(nbytes, TotalBytes());
  CHECK_LE(nbytes, other.TotalBytes());
  CHECK(managed_allocations_.size() &&
        "CopyFrom not supported on unmanaged `external` allocations");
  CHECK(other.managed_allocations_.size() &&
        "CopyFrom not supported on unmanaged `external` allocations");

  if (managed_allocations_.size() == other.managed_allocations_.size()) {
    size_t copied = 0;
    for (size_t i = 0; i < managed_allocations_.size(); ++i) {
      const auto& this_alloc = managed_allocations_[i];
      const auto& other_alloc = other.managed_allocations_[i];

      size_t bytes_to_copy = std::min(nbytes - copied, this_alloc->allocation_nbytes_);
      if (bytes_to_copy == 0) break;

      CHECK_LE(other_alloc->allocation_nbytes_, this_alloc->allocation_nbytes_);

      int status = hexagon_user_dma_1d_sync(this_alloc->data_, other_alloc->data_, bytes_to_copy);
      CHECK_EQ(status, 0);

      copied += bytes_to_copy;
    }
  } else if (managed_allocations_.size() == 1) {
    return other.CopyTo(managed_allocations_[0]->data_, nbytes);
  } else if (other.managed_allocations_.size() == 1) {
    return CopyFrom(other.managed_allocations_[0]->data_, nbytes);
  } else {
    CHECK(false) << "To copy between Hexagon Buffers they must either have the same number of "
                    "dimensions or one of the Hexagon Buffers must have a single dimension.";
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
