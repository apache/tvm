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
#include "hexagon_vtcm_pool.h"

#include "HAP_compute_res.h"
#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonVtcmPool::HexagonVtcmPool() {
  compute_res_attr_t res_info;
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&res_info));

  unsigned int total_block_size;
  unsigned int avail_block_size;
  compute_res_vtcm_page_t total_block_layout;
  compute_res_vtcm_page_t avail_block_layout;

  HEXAGON_SAFE_CALL(compute_resource_query_VTCM(/* application_id = */ 0, &total_block_size,
                                                &total_block_layout, &avail_block_size,
                                                &avail_block_layout));
  DLOG(INFO) << "HexagonVtcmPool total " << total_block_size << " avail " << avail_block_size;
  CHECK(avail_block_size >= (1024 * 1024)) << "Less than 1MB VTCM available";

  // allocate nbytes of vtcm on a single page
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param_v2(&res_info,
                                                           /*vtcm_size = */ total_block_size,
                                                           /*min_page_size = */ 1,
                                                           /*min_vtcm_size = */ avail_block_size));

  // TODO(HWE): Investigate why a non-zero timeout results in
  // hanging, both in the simulator and on hardware.
  context_id_ = HAP_compute_res_acquire(&res_info, /*timeout = */ 0);
  CHECK(context_id_) << "HAP_compute_res_acquire failed to acquire requested VTCM resource.";
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_get_vtcm_ptr_v2(&res_info, &vtcm_data_, &vtcm_size_));
  CHECK(vtcm_data_ != nullptr) << "HAP_compute_res_acquire returned nullptr when allocating VTCM.";
  CHECK(vtcm_size_ >= avail_block_size)
      << "HAP_compute_res_acquire failed to allocate minimum amount of VTCM";
  free_.emplace_back(std::pair<char*, size_t>(static_cast<char*>(vtcm_data_), vtcm_size_));
  // DebugDump();
}

HexagonVtcmPool::~HexagonVtcmPool() { HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_)); }

void* HexagonVtcmPool::Allocate(size_t nbytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  CHECK(!free_.empty()) << "No free VTCM";
  CHECK(nbytes >= 0x80) << "Minimum VTCM alloation must be 128 bytes - nbytes " << nbytes;

  // If this is not aligned on a 2k block, allocate from the end to avoid fragmentation
  if (nbytes & size_t(0x7FF)) {
    DLOG(INFO) << "VTCM nbytes requested: " << nbytes << " allocate from the end";
    auto last_free_entry = free_.end();
    last_free_entry--;
    CHECK(last_free_entry->second >= nbytes)
        << "Not enough contiguous VTCM space at the end to allocate";
    char* ptr = last_free_entry->first + (last_free_entry->second - nbytes);
    allocations_.emplace_back(std::pair<char*, size_t>(ptr, nbytes));
    last_free_entry->second -= nbytes;
    if (last_free_entry->second == 0) {
      free_.erase(last_free_entry);
    }
    // DebugDump();
    return ptr;
  }

  auto entry_to_allocate = free_.begin();
  for (auto it = free_.begin(); it != free_.end(); it++) {
    if ((it->second < entry_to_allocate->second) && (it->second >= nbytes)) {
      entry_to_allocate = it;
      if (entry_to_allocate->second == nbytes) {
        break;
      }
    }
  }
  CHECK(entry_to_allocate->second >= nbytes) << "Not enough contiguous VTCM space to allocate";
  char* ptr = entry_to_allocate->first;
  allocations_.emplace(allocations_.end(), std::pair<char*, size_t>(ptr, nbytes));

  if (entry_to_allocate->second == nbytes) {
    free_.erase(entry_to_allocate);
  } else {
    entry_to_allocate->first = entry_to_allocate->first + nbytes;
    entry_to_allocate->second = entry_to_allocate->second - nbytes;
  }
  // DebugDump();
  return ptr;
}

void HexagonVtcmPool::Free(void* ptr, size_t nbytes) {
  char* ptr_to_free = static_cast<char*>(ptr);
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = std::find_if(allocations_.begin(), allocations_.end(),
                         [&](auto entry) { return entry.first == ptr_to_free; });
  CHECK(it != allocations_.end()) << "Attempted to free a pointer that had not been allocated";
  CHECK(it->second == nbytes) << "Attempted to free a different size than was allocated";
  allocations_.erase(it);

  it = std::lower_bound(free_.begin(), free_.end(), std::pair<char*, size_t>(ptr_to_free, nbytes),
                        [](auto p, auto q) { return p.first <= q.first; });
  if (it == free_.end()) {
    // Insert an entry at the end
    it = free_.emplace(it, std::pair<char*, size_t>(ptr_to_free, nbytes));
  } else {
    CHECK(ptr_to_free != it->first) << "Attempting to free a pointer that was already free";
    CHECK(ptr_to_free + nbytes <= it->first)
        << "free_ is in an inconsistent state, freed block overlaps with next";
    if (ptr_to_free + nbytes == it->first) {
      // Make this entry bigger
      it->first = ptr_to_free;
      it->second += nbytes;
    } else {
      // Insert an entry before this
      it = free_.emplace(it, std::pair<char*, size_t>(ptr_to_free, nbytes));
    }
  }

  // Check for overlap with the previous entry
  if (it != free_.begin()) {
    auto it_prev = it;
    it_prev--;
    CHECK(it_prev->first + it_prev->second <= ptr_to_free)
        << "free_ is in an inconsistent state, freed block overlaps with previous";
    if (it_prev->first + it_prev->second == ptr_to_free) {
      it_prev->second += it->second;
      free_.erase(it);
    }
  }
  // DebugDump();
}

void HexagonVtcmPool::DebugDump() {
  LOG(INFO) << "VTCM list state";
  for (auto entry : allocations_) {
    LOG(INFO) << "VTCM alloc: " << static_cast<void*>(entry.first) << " " << entry.second;
  }
  for (auto entry : free_) {
    LOG(INFO) << "VTCM  free: " << static_cast<void*>(entry.first) << " " << entry.second;
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
