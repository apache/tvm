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

#include "virtual_memory.h"
#include "vta/driver.h"

#include <dmlc/logging.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <utility>
#include <iterator>
#include <unordered_map>

namespace vta {
namespace vmem {

class VirtualMemoryManager {
 public:
  /*! \brief allocate virtual memory for given size */
  void * Allocate(uint64_t size) {
    uint64_t npage = ((size + kPageSize - 1) / kPageSize);
    size_t paged_size = npage * kPageSize;
    void * ptr = malloc(paged_size);
    size_t laddr = reinterpret_cast<size_t>(ptr);
    // search for available virtual memory space
    uint64_t vaddr = VTA_VADDR_BEGIN;
    auto it = page_table_.end();
    if (page_table_.size() > 0) {
      for (it = page_table_.begin(); it != page_table_.end(); it++) {
        if (it == page_table_.begin()) { continue; }
        if (((*it).first - (*std::prev(it)).second) > paged_size) {
          vaddr = (*it).second;
          break;
        }
      }
      if (it == page_table_.end()) {
        vaddr = (*std::prev(it)).second;
      }
    }
    page_table_.insert(it, std::make_pair(vaddr, vaddr + paged_size));
    tlb_[vaddr] = laddr;
    return reinterpret_cast<void*>(vaddr);
  }

  /*! \brief get page table for virtual memory translation. */
  std::vector<uint64_t> GetPageFile() {
    std::vector<uint64_t> vmem_pagefile;
    uint32_t tlb_cnt = 0;
    uint64_t * tlb = new uint64_t[3*tlb_.size()];
    for (auto iter = tlb_.begin(); iter != tlb_.end(); iter++, tlb_cnt++) {
      tlb[tlb_cnt * 3] = (*iter).first;
      uint64_t vend = 0;
      for (auto iter_in = page_table_.begin(); iter_in != page_table_.end(); iter_in++) {
        if ((*iter_in).first == (*iter).first) { vend = (*iter_in).second; break; }
      }
      tlb[tlb_cnt * 3 + 1] = vend;
      tlb[tlb_cnt * 3 + 2] = (*iter).second;
      vmem_pagefile.push_back(tlb[tlb_cnt * 3 + 0]);
      vmem_pagefile.push_back(tlb[tlb_cnt * 3 + 1]);
      vmem_pagefile.push_back(tlb[tlb_cnt * 3 + 2]);
    }
    delete [] tlb;
    return vmem_pagefile;
  }

  /*! \brief release virtual memory for pointer */
  void Release(void * ptr) {
    uint64_t src = reinterpret_cast<uint64_t>(ptr);
    auto it = page_table_.begin();
    for (; it != page_table_.end(); it++) {
      if (((*it).first <= src) && ((*it).second > src)) { break; }
    }
    CHECK(it != page_table_.end());
    uint64_t * laddr = reinterpret_cast<uint64_t*>(tlb_[(*it).first]);
    delete [] laddr;
    page_table_.erase(it);
    tlb_.erase((*it).first);
  }

  /*! \brief copy virtual memory from host */
  void MemCopyFromHost(void * dstptr, const void * src, uint64_t size) {
    // get logical address from virtual address
    size_t dst = reinterpret_cast<size_t>(dstptr);
    auto it = page_table_.begin();
    for (; it != page_table_.end(); it++) {
      if (((*it).first <= dst) && ((*it).second > dst)) { break; }
    }
    CHECK(it != page_table_.end());
    size_t offset = dst - (*it).first;
    char * laddr = reinterpret_cast<char*>(tlb_[(*it).first]);
    // copy content from src to logic address
    memcpy(laddr + offset, src, size);
  }

  /*! \brief copy virtual memory to host */
  void MemCopyToHost(void * dst, const void * srcptr, uint64_t size) {
    // get logical address from virtual address
    size_t src = reinterpret_cast<size_t>(srcptr);
    auto it = page_table_.begin();
    for (; it != page_table_.end(); it++) {
      if (((*it).first <= src) && ((*it).second > src)) { break; }
    }
    CHECK(it != page_table_.end());
    size_t offset = src - (*it).first;
    char * laddr = reinterpret_cast<char*>(tlb_[(*it).first]);
    // copy content from logic address to dst
    memcpy(dst, laddr + offset, size);
  }

  /*! \brief get logical address from virtual memory */
  void * GetLogicalAddr(uint64_t src) {
    if (src == 0) { return 0; }
    auto it = page_table_.begin();
    for (; it != page_table_.end(); it++) {
      if (((*it).first <= src) && ((*it).second > src)) { break; }
    }
    CHECK(it != page_table_.end());
    return reinterpret_cast<void*>(tlb_[(*it).first]);
  }

  /*! \brief get global handler of the instance */
  static VirtualMemoryManager* Global() {
    static VirtualMemoryManager inst;
    return &inst;
  }

 private:
  // page size, also the maximum allocable size 16 K
  static const uint64_t kPageSize = VTA_PAGE_BYTES;
  /*! \brief page table */
  std::list<std::pair<uint64_t, uint64_t> > page_table_;
  /*! \brief translation lookaside buffer */
  std::unordered_map<uint64_t, size_t> tlb_;
};

}  // namespace vmem
}  // namespace vta


void * vmalloc(uint64_t size) {
  return vta::vmem::VirtualMemoryManager::Global()->Allocate(size);
}

void vfree(void * ptr) {
  vta::vmem::VirtualMemoryManager::Global()->Release(ptr);
}

void vmemcpy(void * dst, const void * src, uint64_t size, VMemCopyType dir) {
  auto * mgr = vta::vmem::VirtualMemoryManager::Global();
  if (kVirtualMemCopyFromHost == dir) {
    mgr->MemCopyFromHost(dst, src, size);
  } else {
    mgr->MemCopyToHost(dst, src, size);
  }
}

void * vmem_get_log_addr(uint64_t vaddr) {
  return vta::vmem::VirtualMemoryManager::Global()->GetLogicalAddr(vaddr);
}

std::vector<uint64_t> vmem_get_pagefile() {
  return vta::vmem::VirtualMemoryManager::Global()->GetPageFile();
}
