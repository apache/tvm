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

#include "hexagon_common.h"
#include "hexagon_device_api.h"
#include "qurt_memory.h"

namespace tvm {
namespace runtime {
namespace hexagon {

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

    // The heap used by malloc on Hexagon is always mapped as cacheable. The heap manager may not
    // perform cache invalidation on a prior memory free. So, a subsequent memory allocation request
    // to the heap manager may allocate memory that resides in part or in full in the cache. Hence,
    // we must invalidate the allocation from the cache to ensure that DMA with cache bypass enabled
    // will function properly. DMA with cache bypass enabled assumes that HexagonBuffer objects are
    // not cached unless explicitly modified by the primfunc. We must invalidate after malloc to
    // uphold this assumption.
    qurt_mem_cache_clean(reinterpret_cast<qurt_addr_t>(data_), nbytes, QURT_MEM_CACHE_INVALIDATE,
                         QURT_MEM_DCACHE);
  }
  ~DDRAllocation() { free(data_); }
};

struct VTCMAllocation : public Allocation {
  VTCMAllocation(size_t nbytes, size_t alignment) : Allocation(nbytes, alignment) {
    // For simplicity, the current VTCM dynamic pool supports the following alignments: less than
    // or equal to 128 (0x80), and 2k (0x800)
    CHECK((alignment <= 0x80) || (alignment == 0x800))
        << "VTCMAllocation called for invalid alignment " << alignment;

    if (alignment == 0x800) {
      // Adjust size to be a multiple of 2k so that we will allocate from the front of the pool.
      nbytes = (nbytes + 0x7ff) & -0x800;
    } else if (alignment <= 0x80) {
      // Adjust size to be a multiple of 128 so that we will allocate from the back of the pool
      // in 128 byte increments.
      nbytes = (nbytes + 0x7f) & -0x80;
    }
    if (allocation_nbytes_ != nbytes) {
      DLOG(INFO) << "VTCMAllocation size adjusted for alignment " << allocation_nbytes_ << " to "
                 << nbytes;
      allocation_nbytes_ = nbytes;
    }
    data_ = HexagonDeviceAPI::Global()->VtcmPool()->Allocate(allocation_nbytes_);
    DLOG(INFO) << "VTCMAllocation " << data_ << " " << allocation_nbytes_ << " " << alignment;
  }
  ~VTCMAllocation() {
    DLOG(INFO) << "~VTCMAllocation " << data_ << " " << allocation_nbytes_;
    HexagonDeviceAPI::Global()->VtcmPool()->Free(data_, allocation_nbytes_);
    data_ = nullptr;
  }
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
  }
}

HexagonBuffer::StorageScope HexagonBuffer::GetStorageScope() const { return storage_scope_; }

void HexagonBuffer::SetStorageScope(Optional<String> scope) {
  const std::string s = scope.value_or("global");

  if (s == "global") {
    storage_scope_ = StorageScope::kDDR;
  } else if (s == "global.ddr") {
    storage_scope_ = StorageScope::kDDR;
  } else if (s == "global.vtcm") {
    storage_scope_ = StorageScope::kVTCM;
  } else {
    CHECK(false) << "Encountered unknown HexagonBuffer storage scope: " << std::string(s);
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
                                        size_t bytes_to_copy, bool src_is_hexbuff,
                                        bool dest_is_hexbuff) {
  // First, determine all copies that do not cross boundaries in
  // either source or destination region.
  auto micro_copies = BufferSet::MemoryCopies(dest, src, bytes_to_copy);

  // If regions are contiguously allocated, we can reduce the number
  // of copies required by merging adjacent copies.
  auto macro_copies = MemoryCopy::MergeAdjacent(std::move(micro_copies));

  // Finally, do the memory copies.
  for (const auto& copy : macro_copies) {
    // if src is a HexagonBuffer, invalidate it before the memcpy
    if (src_is_hexbuff) {
      qurt_mem_cache_clean(reinterpret_cast<qurt_addr_t>(copy.src), copy.num_bytes,
                           QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }

    // TODO(HWE): Switch to ION Buffer to avoid need for memcpy and potentially lighten or alleviate
    // the burden of cache invalidation in this code
    memcpy(copy.dest, copy.src, copy.num_bytes);

    // if dest is a HexagonBuffer, flush it after the memcpy
    if (dest_is_hexbuff) {
      qurt_mem_cache_clean(reinterpret_cast<qurt_addr_t>(copy.dest), copy.num_bytes,
                           QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
  }
}

void HexagonBuffer::CopyTo(void* data, size_t nbytes) const {
  BufferSet src(allocations_.data(), allocations_.size(), nbytes_per_allocation_);
  BufferSet dest(&data, 1, nbytes);

  hexagon_buffer_copy_across_regions(dest, src, nbytes, true /* src_is_hexbuff */,
                                     false /* dest_is_hexbuff */);
}

void HexagonBuffer::CopyFrom(void* data, size_t nbytes) {
  BufferSet src(&data, 1, nbytes);
  BufferSet dest(allocations_.data(), allocations_.size(), nbytes_per_allocation_);

  hexagon_buffer_copy_across_regions(dest, src, nbytes, false /* src_is_hexbuff */,
                                     true /* dest_is_hexbuff */);
}

void HexagonBuffer::CopyFrom(const HexagonBuffer& other, size_t nbytes) {
  BufferSet src(other.allocations_.data(), other.allocations_.size(), other.nbytes_per_allocation_);
  BufferSet dest(allocations_.data(), allocations_.size(), nbytes_per_allocation_);

  hexagon_buffer_copy_across_regions(dest, src, nbytes, true /* src_is_hexbuff */,
                                     true /* dest_is_hexbuff */);
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
