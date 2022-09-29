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
#include "hexagon_vtcm.h"

#include "HAP_compute_res.h"
#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

VtcmMemoryManager::VtcmMemoryManager() {
    compute_res_attr_t res_info;
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_init(&res_info));

    // allocate nbytes of vtcm on a single page
    HEXAGON_SAFE_CALL(HAP_compute_res_attr_set_vtcm_param(&res_info, /*vtcm_size = */ vtcm_size_,
                                                          /*b_single_page = */ 0));

    // TODO(HWE): Investigate why a non-zero timeout results in
    // hanging, both in the simulator and on hardware.
    context_id_ = HAP_compute_res_acquire(&res_info, /*timeout = */ 0);

    if (context_id_) {
      vtcm_data_ = HAP_compute_res_attr_get_vtcm_ptr(&res_info);
      if (!vtcm_data_) {
        LOG(ERROR) << "ERROR: HAP_compute_res_acquire returned nullptr when allocating VTCM.";
        HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
        return;
      }
    } else {
      LOG(FATAL) << "FATAL: HAP_compute_res_acquire failed to acquire requested VTCM resource.";
      throw std::runtime_error(
          "HAP_compute_res_acquire failed to acquire requested VTCM resource.");
    }

    free_.emplace_back(std::pair<char*, size_t>((char*)vtcm_data_, vtcm_size_));
}

VtcmMemoryManager::~VtcmMemoryManager() {
    HEXAGON_SAFE_CALL(HAP_compute_res_release(context_id_));
}

void* VtcmMemoryManager::Allocate(size_t nbytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG(INFO) << "nbytes requested: " << nbytes;

  if (nbytes & size_t(0x7FF)) {
    nbytes = nbytes & ~(size_t(0x7FF)) + 0x800;
    LOG(INFO) << "nbytes requested (UPDATED): " << nbytes; // jlsfix - is this right?
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

  return (void*)ptr;
}

void VtcmMemoryManager::Free(void* ptr, size_t nbytes) {
  char* ptr_to_free = (char*)ptr;
  std::lock_guard<std::mutex> lock(mutex_);
  bool found_allocation_entry = false;
  for (auto it = allocations_.begin(); it != allocations_.end(); it++)
  {
    if (ptr_to_free == it->first) {
      // jlsfix CHECK(it->second == nbytes) << "Attempted to free a different size than was allocated";
      nbytes = it->second; // jlsfix
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
        free_.insert(it, std::pair<char*, size_t>(ptr_to_free, nbytes));
        it--;
      }
      break;
    }
  }

  if (it == free_.end()) {
    // Insert an entry before this
    free_.insert(it, std::pair<char*, size_t>(ptr_to_free, nbytes));
    it--;
  }

  // Check for overlap with the previous entry
  auto it_prev = it; it_prev--;
  CHECK(it_prev->first + it_prev->second > ptr_to_free) << "free_ is in an inconsistent state, freed block overlaps with previous";
  if (it_prev->first + it_prev->second == ptr_to_free) {
    it_prev->second += nbytes;
    free_.erase(it);
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
