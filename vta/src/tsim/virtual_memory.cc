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

#include <cstdint>
#include <cstdlib>
#include <list>
#include <utility>
#include <iterator>
#include <unordered_map>

/*! page size of virtual address */
#ifndef VTA_TSIM_VM_PAGE_SIZE
#define VTA_TSIM_VM_PAGE_SIZE (4096)
#endif  // VTA_TSIM_VM_PAGE_SIZE

/*! starting point of the virtual address system */
#ifndef VTA_TSIM_VM_ADDR_BEGIN
#define VTA_TSIM_VM_ADDR_BEGIN (0x40000000)
#endif  // VTA_TSIM_VM_ADDR_BEGIN

namespace vta {
namespace tsim {
    
static const uint64_t kPageSize = VTA_TSIM_VM_PAGE_SIZE;
static const uint64_t kVirtualMemoryOffset = VTA_TSIM_VM_ADDR_BEGIN;

class VirtualMemoryManager {
public:
  /*! \brief page table */
  std::list<std::pair<uint64_t, uint64_t> > page_table;
  /*! \brief translation lookaside buffer */
  std::unordered_map<uint64_t, size_t> tlb;
  
  void * Allocate(uint64_t sz) {
    uint64_t size = ((sz + kPageSize - 1) / kPageSize) * kPageSize;
    void * ptr = malloc(size);
    size_t laddr = reinterpret_cast<size_t>(ptr);
    // search for available virtual memory space
    uint64_t vaddr = kVirtualMemoryOffset;
    auto it = page_table.begin();
    if (((*it).first - kVirtualMemoryOffset) < size) {
      it++;
      for (; it != page_table.end(); it++) {
        if (((*it).first - (*std::prev(it)).second) > size) {
          vaddr = (*std::prev(it)).second;
        }
      }
      if (it == page_table.end()) {
        vaddr = (*std::prev(it)).second;
      }
    }
    page_table.push_back(std::make_pair(vaddr, vaddr + size));
    tlb[vaddr] = laddr;
    return reinterpret_cast<void*>(vaddr);
  }
  void Release(void * ptr) {
    (void)ptr;
  }
  static VirtualMemoryManager* Global() {
    static VirtualMemoryManager inst;
    return &inst;
  }
};
  
}  // namespace tsim
}  // namespace vta


void * vmalloc(uint64_t size) {
  return vta::tsim::VirtualMemoryManager::Global()->Allocate(size);
}

void vfree(void * ptr) {
  vta::tsim::VirtualMemoryManager::Global()->Release(ptr);
}
