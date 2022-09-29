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

  // TODO(HWE): get the max  and min size programmatically
  const unsigned int max_size = 4*1024*1024;
  const unsigned int min_size = 1024*1024;

  // allocate nbytes of vtcm on a single page
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param_v2(&res_info,
                                                        /*vtcm_size = */ max_size,
                                                        /*min_page_size = */ 0,
                                                        /*min_vtcm_size = */ min_size));

  // TODO(HWE): Investigate why a non-zero timeout results in
  // hanging, both in the simulator and on hardware.
  context_id_ = HAP_compute_res_acquire(&res_info, /*timeout = */ 0);
  CHECK(context_id_) << "HAP_compute_res_acquire failed to acquire requested VTCM resource.";
  HEXAGON_SAFE_CALL(HAP_compute_res_attr_get_vtcm_ptr_v2(&res_info, &vtcm_data_, &vtcm_size_));
  CHECK(vtcm_data_ != nullptr) << "HAP_compute_res_acquire returned nullptr when allocating VTCM.";
  CHECK(vtcm_size_ >= min_size)
    << "HAP_compute_res_acquire failed to allocate minimum amount of VTCM";
  free_.emplace_back(std::pair<char*, size_t>((char*)vtcm_data_, vtcm_size_));

  DebugDump();
}

HexagonVtcmPool::~HexagonVtcmPool() {
  HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
}

void* HexagonVtcmPool::Allocate(size_t nbytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (nbytes & size_t(0x7FF)) {
    size_t nbytes_requested = nbytes;
    nbytes = nbytes >> 11;
    nbytes = nbytes << 11;
    nbytes += 0x800;
    LOG(INFO) << "nbytes requested: " << nbytes_requested << " nbytes (UPDATED): " << nbytes;
  }

  CHECK(!free_.empty()) << "No free VTCM";

  std::pair<char*, size_t>& entry_to_allocate = free_.front();
  for(auto entry : free_) {
    if ((entry.second < entry_to_allocate.second) && (entry.second >= nbytes))
    {
      entry_to_allocate = entry;
      if (entry_to_allocate.second == nbytes) {
        break;
      }
    }
  }
  CHECK(entry_to_allocate.second >= nbytes) << "Not enough contiguous VTCM space to allocate";
  char* ptr = entry_to_allocate.first;
  allocations_.emplace(allocations_.end(), std::pair<char*, size_t>(ptr, nbytes));

  for (auto it = free_.begin(); it != free_.end(); it++) {
    if (ptr == it->first) {
      if (it->second == nbytes) {
        free_.erase(it);
      } else {
        it->first = it->first + nbytes;
        it->second = it->second - nbytes;
      }
      break;
    }
  }

  DebugDump();

  return (void*)ptr;
}

void HexagonVtcmPool::Free(void* ptr, size_t nbytes) {
  char* ptr_to_free = (char*)ptr;
  std::lock_guard<std::mutex> lock(mutex_);

  if (nbytes & size_t(0x7FF)) {
    size_t nbytes_requested = nbytes;
    nbytes = nbytes >> 11;
    nbytes = nbytes << 11;
    nbytes += 0x800;
    LOG(INFO) << "nbytes requested: " << nbytes_requested << " nbytes (UPDATED): " << nbytes;
  }

  bool found_allocation_entry = false;
  for (auto it = allocations_.begin(); it != allocations_.end(); it++)
  {
    if (ptr_to_free == it->first) {
      CHECK(it->second == nbytes) << "Attempted to free a different size than was allocated";
      allocations_.erase(it);
      found_allocation_entry = true;
      break;
    }
  }
  CHECK(found_allocation_entry) << "Attempted to free a pointer that had not been allocated";

  auto it = free_.begin();
  for ( ; it != free_.end(); it++) {
    CHECK(ptr_to_free != it->first) << "Attempting to free a pointer that was already free";
    if (ptr_to_free < it->first) {
      CHECK(ptr_to_free + nbytes <= it->first) << "free_ is in an inconsistent state, freed block overlaps with next";
      if (ptr_to_free + nbytes == it->first) {
        // Make this entry bigger
        it->first = ptr_to_free;
        it->second += nbytes;
      } else {
        // Insert an entry before this
        it = free_.emplace(it, std::pair<char*, size_t>(ptr_to_free, nbytes));
      }
      break;
    }
  }

  if (it == free_.end()) {
    // Insert an entry before this
    it = free_.emplace(it, std::pair<char*, size_t>(ptr_to_free, nbytes));
  }

  // Check for overlap with the previous entry
  if (it != free_.begin()) {
    auto it_prev = it; it_prev--;
    CHECK(it_prev->first + it_prev->second <= ptr_to_free) << "free_ is in an inconsistent state, freed block overlaps with previous";
    if (it_prev->first + it_prev->second == ptr_to_free) {
      it_prev->second += nbytes;
      free_.erase(it);
    }
  }

  DebugDump();
}

void HexagonVtcmPool::DebugDump()
{
  LOG(INFO) << "VTCM list state";
  for (auto it = allocations_.begin(); it != allocations_.end(); it++) {
    LOG(INFO) << "VTCM alloc: " << (void*)it->first << " " << it->second;
  }

  for (auto it = free_.begin(); it != free_.end(); it++) {
    LOG(INFO) << "VTCM free:  " << (void*)it->first << " " << it->second;
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
