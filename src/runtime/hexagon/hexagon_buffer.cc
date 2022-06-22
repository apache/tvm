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

#include <algorithm>
#include <string>
#include <utility>

#include "HAP_compute_res.h"
#include "hexagon_common.h"

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
    int ret = posix_memalign(&data_, alignment, nbytes);
    CHECK_EQ(ret, 0);
  }
  ~DDRAllocation() { free(data_); }
};

struct VTCMAllocation : public Allocation {
  VTCMAllocation(size_t nbytes, size_t alignment) : Allocation(nbytes, alignment) {
    compute_res_attr_t res_info;
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&res_info));

    // allocate nbytes of vtcm on a single page
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param(&res_info, /*vtcm_size = */ nbytes,
                                                          /*b_single_page = */ 1));

    // TODO(HWE): Investigate why a non-zero timeout results in
    // hanging, both in the simulator and on hardware.
    context_id_ = HAP_compute_res_acquire(&res_info, /*timeout = */ 0);

    if (context_id_) {
      data_ = HAP_compute_res_attr_get_vtcm_ptr(&res_info);
      if (!data_) {
        LOG(ERROR) << "ERROR: Allocated VTCM ptr is null.";
        HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
        return;
      }
    } else {
      LOG(ERROR) << "ERROR: Unable to acquire requeisted resource.";
      return;
    }
  }
  ~VTCMAllocation() {
    HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
    data_ = nullptr;
  }
  unsigned int context_id_{0};
};

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

  size_t nbytes_aligned = ((nbytes + (alignment - 1)) / alignment) * alignment;
  size_t nbytes_monolithic = nallocs * nbytes_aligned;

  std::unique_ptr<Allocation> alloca = nullptr;
  if (GetStorageScope() == StorageScope::kDDR) {
    alloca = Allocator<StorageScope::kDDR>(nbytes_monolithic, alignment);
  } else if (GetStorageScope() == StorageScope::kVTCM) {
    alloca = Allocator<StorageScope::kVTCM>(nbytes_monolithic, alignment);
  }
  CHECK(alloca) << "could not create allocation";

  for (size_t i = 0; i < nallocs; ++i) {
    void* alloc_offset = static_cast<unsigned char*>(alloca->data_) + i * nbytes_aligned;
    allocations_.push_back(alloc_offset);
  }

  managed_allocations_.push_back(std::move(alloca));
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

std::vector<MemoryCopy> BufferSet::MemoryCopies(const BufferSet& dest, const BufferSet& src,
                                                size_t bytes_to_copy) {
  CHECK_LE(bytes_to_copy, src.TotalBytes());
  CHECK_LE(bytes_to_copy, dest.TotalBytes());

  auto pointer_to = [](const BufferSet& buf, size_t region_i, size_t byte_i) -> void* {
    void* region = buf.buffers[region_i];
    return static_cast<unsigned char*>(region) + byte_i;
  };

  size_t num_src_regions = (bytes_to_copy + src.region_size_bytes - 1) / src.region_size_bytes;

  // First, determine all copies that do not cross boundaries in
  // either source or destination region.  This requires two loops, as
  // a single source region may overlap one or more destination
  // regions, and vice versa.
  std::vector<MemoryCopy> micro_copies;
  for (size_t src_i = 0; src_i < num_src_regions; src_i++) {
    size_t src_region_begin = src_i * src.region_size_bytes;
    size_t src_region_end = std::min((src_i + 1) * src.region_size_bytes, bytes_to_copy);

    size_t dest_i_begin = src_region_begin / dest.region_size_bytes;
    size_t dest_i_end = (src_region_end - 1) / dest.region_size_bytes + 1;
    for (size_t dest_i = dest_i_begin; dest_i < dest_i_end; dest_i++) {
      size_t offset_begin = std::max(src_region_begin, dest_i * dest.region_size_bytes);
      size_t offset_end = std::min(src_region_end, (dest_i + 1) * dest.region_size_bytes);

      size_t num_bytes = offset_end - offset_begin;
      void* src_ptr = pointer_to(src, src_i, offset_begin % src.region_size_bytes);
      void* dest_ptr = pointer_to(dest, dest_i, offset_begin % dest.region_size_bytes);
      micro_copies.push_back(MemoryCopy(dest_ptr, src_ptr, num_bytes));
    }
  }

  return micro_copies;
}

std::vector<MemoryCopy> MemoryCopy::MergeAdjacent(std::vector<MemoryCopy> micro_copies) {
  std::sort(micro_copies.begin(), micro_copies.end(),
            [](const MemoryCopy& a, const MemoryCopy& b) { return a.src < b.src; });

  std::vector<MemoryCopy> macro_copies;
  for (const auto& copy : micro_copies) {
    if (macro_copies.size() && macro_copies.back().IsDirectlyBefore(copy)) {
      macro_copies.back().num_bytes += copy.num_bytes;
    } else {
      macro_copies.push_back(copy);
    }
  }

  return macro_copies;
}

void hexagon_buffer_copy_across_regions(const BufferSet& dest, const BufferSet& src,
                                        size_t bytes_to_copy) {
  // First, determine all copies that do not cross boundaries in
  // either source or destination region.
  auto micro_copies = BufferSet::MemoryCopies(dest, src, bytes_to_copy);

  // If regions are contiguously allocated, we can reduce the number
  // of copies required by merging adjacent copies.
  auto macro_copies = MemoryCopy::MergeAdjacent(std::move(micro_copies));

  // Finally, do the memory copies.
  for (const auto& copy : macro_copies) {
    int error_code = hexagon_user_dma_1d_sync(copy.dest, copy.src, copy.num_bytes);
    CHECK_EQ(error_code, 0);
  }
}

void HexagonBuffer::CopyTo(void* data, size_t nbytes) const {
  BufferSet src(allocations_.data(), allocations_.size(), nbytes_per_allocation_);
  BufferSet dest(&data, 1, nbytes);

  hexagon_buffer_copy_across_regions(dest, src, nbytes);
}

void HexagonBuffer::CopyFrom(void* data, size_t nbytes) {
  BufferSet src(&data, 1, nbytes);
  BufferSet dest(allocations_.data(), allocations_.size(), nbytes_per_allocation_);

  hexagon_buffer_copy_across_regions(dest, src, nbytes);
}

void HexagonBuffer::CopyFrom(const HexagonBuffer& other, size_t nbytes) {
  BufferSet src(other.allocations_.data(), other.allocations_.size(), other.nbytes_per_allocation_);
  BufferSet dest(allocations_.data(), allocations_.size(), nbytes_per_allocation_);

  hexagon_buffer_copy_across_regions(dest, src, nbytes);
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
