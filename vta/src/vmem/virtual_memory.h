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

/*!
 *  Copyright (c) 2019 by Contributors
 * \file virtual_memory.h
 * \brief The virtual memory manager for device simulation
 */

#ifndef VTA_VMEM_VIRTUAL_MEMORY_H_
#define VTA_VMEM_VIRTUAL_MEMORY_H_

#include <vta/driver.h>
#include <cstdint>
#include <type_traits>
#include <mutex>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>

enum VMemCopyType {
  kVirtualMemCopyFromHost = 0,
  kVirtualMemCopyToHost = 1
};

namespace vta {
namespace vmem {

/*!
 * \brief DRAM memory manager
 *  Implements simple paging to allow physical address translation.
 */
class VirtualMemoryManager {
 public:
  /*!
   * \brief Get virtual address given physical address.
   * \param phy_addr The simulator phyiscal address.
   * \return The true virtual address;
   */
  void* GetAddr(uint64_t phy_addr);
  /*!
   * \brief Get physical address
   * \param buf The virtual address.
   * \return The true physical address;
   */
  vta_phy_addr_t GetPhyAddr(void* buf);
  /*!
   * \brief Allocate memory from manager
   * \param size The size of memory
   * \return The virtual address
   */
  void* Alloc(size_t size);
  /*!
   * \brief Free the memory.
   * \param size The size of memory
   * \return The virtual address
   */
  void Free(void* data);
  void MemCopyFromHost(void* dst, const void * src, size_t size);
  void MemCopyToHost(void* dst, const void * src, size_t size);
  static VirtualMemoryManager* Global();

 private:
  // The bits in page table
  static constexpr vta_phy_addr_t kPageBits = VTA_PAGE_BITS;
  // page size, also the maximum allocable size 16 K
  static constexpr vta_phy_addr_t kPageSize = VTA_PAGE_BYTES;
  /*! \brief A page in the DRAM */
  struct Page {
    /*! \brief Data Type */
    using DType = typename std::aligned_storage<kPageSize, 256>::type;
    /*! \brief Start location in page table */
    size_t ptable_begin;
    /*! \brief The total number of pages */
    size_t num_pages;
    /*! \brief Data */
    DType* data{nullptr};
    // construct a new page
    explicit Page(size_t ptable_begin, size_t num_pages)
        : ptable_begin(ptable_begin), num_pages(num_pages) {
      data = new DType[num_pages];
    }
    ~Page() {
      delete [] data;
    }
  };
  // Internal lock
  std::mutex mutex_;
  // Physical address -> page
  std::vector<Page*> ptable_;
  // virtual addres -> page
  std::unordered_map<void*, std::unique_ptr<Page> > pmap_;
  // Free map
  std::multimap<size_t, Page*> free_map_;
};


}  // namespace vmem
}  // namespace vta

/*!
 * \brief virtual memory based memory allocation
 */
void * vmalloc(uint64_t size);

/*!
 * \brief virtual memory based memory release
 */
void vfree(void * ptr);

/*!
 * \brief memory copy between virtual and logical
 */
void vmemcpy(void * dst, const void * src, uint64_t size, VMemCopyType dir);

/*!
 * \brief map virtual address to logical address
 */
void * vmem_get_addr(uint64_t vaddr);

#endif /* VTA_VMEM_VIRTUAL_MEMORY_H_ */
