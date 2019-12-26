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
 * \file virtual_memory.cc
 * \brief Thread-safe virtal memory manager
 */

#include "virtual_memory.h"

#include <dmlc/logging.h>
#include <vta/driver.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <utility>
#include <iterator>
#include <unordered_map>
#include <map>
#include <mutex>

namespace vta {
namespace vmem {

/*!
 * \brief Get virtual address given physical address.
 * \param phy_addr The simulator phyiscal address.
 * \return The true virtual address;
 */
void* VirtualMemoryManager::GetAddr(uint64_t phy_addr) {
  CHECK_NE(phy_addr, 0)
      << "trying to get address that is nullptr";
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t loc = (phy_addr >> kPageBits) - 1;
  CHECK_LT(loc, ptable_.size())
      << "phy_addr=" << phy_addr;
  Page* p = ptable_[loc];
  CHECK(p != nullptr);
  size_t offset = (loc - p->ptable_begin) << kPageBits;
  offset += phy_addr & (kPageSize - 1);
  return reinterpret_cast<char*>(p->data) + offset;
}

/*!
 * \brief Get physical address
 * \param buf The virtual address.
 * \return The true physical address;
 */
vta_phy_addr_t VirtualMemoryManager::GetPhyAddr(void* buf) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pmap_.find(buf);
  uint64_t offset = 0;
  if (it == pmap_.end()) {
    for (it = pmap_.begin(); it != pmap_.end(); it++) {
      uint64_t bytes = it->second->num_pages << kPageBits;
      if ((buf >= it->first) && (buf < static_cast<char*>(it->first) + bytes)) {
        offset = static_cast<char*>(buf) - static_cast<char*>(it->first);
        break;
      }
    }
    CHECK(it != pmap_.end());
  }
  Page* p = it->second.get();
  return ((p->ptable_begin + 1) << kPageBits) + offset;
}

/*!
 * \brief Allocate memory from manager
 * \param size The size of memory
 * \return The virtual address
 */
void* VirtualMemoryManager::Alloc(size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t npage = (size + kPageSize - 1) / kPageSize;
  auto it = free_map_.lower_bound(npage);
  if (it != free_map_.end()) {
    Page* p = it->second;
    free_map_.erase(it);
    return p->data;
  }
  size_t start = ptable_.size();
  std::unique_ptr<Page> p(new Page(start, npage));
  // insert page entry
  ptable_.resize(start + npage, p.get());
  void* data = p->data;
  pmap_[data] = std::move(p);
  return data;
}

/*!
 * \brief Free the memory.
 * \param size The size of memory
 * \return The virtual address
 */
void VirtualMemoryManager::Free(void* data) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pmap_.size() == 0) return;
  auto it = pmap_.find(data);
  CHECK(it != pmap_.end());
  Page* p = it->second.get();
  free_map_.insert(std::make_pair(p->num_pages, p));
}

/*!
 * \brief Copy from the host memory to device memory (virtual).
 * \param dst The device memory address (virtual)
 * \param src The host memory address
 * \param size The size of memory
 */
void VirtualMemoryManager::MemCopyFromHost(void* dst, const void * src, size_t size) {
  void * addr = this->GetAddr(reinterpret_cast<uint64_t>(dst));
  memcpy(addr, src, size);
}

/*!
 * \brief Copy from the device memory (virtual) to host memory.
 * \param dst The host memory address
 * \param src The device memory address (virtual)
 * \param size The size of memory
 */
void VirtualMemoryManager::MemCopyToHost(void* dst, const void * src, size_t size) {
  void * addr = this->GetAddr(reinterpret_cast<uint64_t>(src));
  memcpy(dst, addr, size);
}

VirtualMemoryManager* VirtualMemoryManager::Global() {
  static VirtualMemoryManager inst;
  return &inst;
}

}  // namespace vmem
}  // namespace vta
