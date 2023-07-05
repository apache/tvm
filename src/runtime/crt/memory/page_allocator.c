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

// LINT_C_FILE

/*!
 * \file memory.c
 * \brief Virtual memory manager
 *
 * To maximize portability, thread-safe feature has been dropped for now.
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/internal/memory/page_allocator.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/platform.h>

// construct a new page
Page PageCreate(uint8_t* memory_pool, size_t page_size_bytes, tvm_index_t ptable_begin,
                tvm_index_t num_pages) {
  Page page;
  page.ptable_begin = ptable_begin;
  page.num_pages = num_pages;
  page.data = memory_pool + ptable_begin * page_size_bytes;
  return page;
}

void PageTable_Resize(struct PageTable* ptable, size_t new_size, Page* page) {
  CHECK_LE(ptable->num_pages, new_size, "size value (%zu) is smaller than expected (%zu).",
           new_size, ptable->num_pages);
  for (uint32_t idx = ptable->num_pages; idx < new_size; idx++) {
    ptable->page[idx] = *page;
  }
  ptable->num_pages = new_size;
}

void TLB_Set(TLB* tlb, uint8_t* data, Page* page) {
  PageEntry* entry = tlb->find(tlb, data);
  if (entry == 0) {
    tlb->entries[tlb->num_pages].addr = data;
    tlb->entries[tlb->num_pages].page = *page;
    tlb->num_pages++;
  } else {
    entry->addr = data;
    entry->page = *page;
  }
}

PageEntry* TLB_Find(TLB* tlb, uint8_t* data) {
  PageEntry* entry = 0;
  for (uint32_t idx = 0; idx < tlb->num_pages; idx++) {
    if (tlb->entries[idx].addr == data) {
      entry = tlb->entries + idx;
      break;
    }
  }
  return entry;
}

IndexedEntry* MultiMap_LowerBound(struct MultiMap* map, uint32_t npage) {
  IndexedEntry* entry = 0;
  for (uint32_t idx = 0; idx < map->num_entries; idx++) {
    if (map->entries[idx].index >= npage) {
      entry = map->entries + idx;
      break;
    }
  }
  return entry;
}

IndexedEntry* MultiMap_End(struct MultiMap* map) {
  IndexedEntry* entry = 0;
  return entry;
}

void MultiMap_Erase(struct MultiMap* map, IndexedEntry* entry) {
  for (uint32_t idx = 0; idx < map->num_entries; idx++) {
    if ((map->entries + idx) == entry) {
      // NOTE: do not use memcpy due to overlap.
      for (uint32_t src_idx = idx + 1; src_idx < map->num_entries; src_idx++) {
        map->entries[src_idx - 1] = map->entries[src_idx];
      }
      map->num_entries--;
      break;
    }
  }
}

void MultiMap_Insert(struct MultiMap* map, uint32_t npage, Page* p) {
  CHECK_LE(map->num_entries + 1, map->max_entries, "invalid number of free pages.");
  for (uint32_t idx = map->num_entries; idx < (map->num_entries + npage); idx++) {
    map->entries[map->num_entries].index = npage;
    map->entries[map->num_entries].page = *p;
  }
  map->num_entries++;
}

/*!
 * \brief Allocate memory from manager
 * \param size The size of memory
 * \return The virtual address
 */
tvm_crt_error_t PageMemoryManager_Allocate(MemoryManagerInterface* interface, size_t num_bytes,
                                           DLDevice dev, void** out_ptr) {
  MemoryManager* mgr = (MemoryManager*)interface;

  *out_ptr = 0;
  PageTable* ptable = &(mgr->ptable);
  tvm_index_t npage = (num_bytes + ptable->page_size_bytes - 1) / ptable->page_size_bytes;

  MultiMap* free_map = &(mgr->free_map);
  IndexedEntry* it = free_map->lower_bound(free_map, npage);
  tvm_index_t start = 0;
  if (it != free_map->end(free_map)) {
    Page p = it->page;
    free_map->erase(free_map, it);
    *out_ptr = p.data;
    start = p.ptable_begin;
    npage = p.num_pages;
  } else {
    start = ptable->num_pages;
    if ((unsigned)(start + npage) > ptable->max_pages) {
#if TVM_CRT_DEBUG > 1
      TVMLogf("insufficient memory, start=%" PRId32 ", npage=%" PRId32 ", total=%" PRId32 " / %zu",
              (int32_t)start, (int32_t)npage, (int32_t)(start + npage), mgr->pmap.max_pages);
#endif
      return kTvmErrorPlatformNoMemory;
    }
    /* insert page entry */
    Page p = PageCreate(ptable->memory_pool, ptable->page_size_bytes, start, npage);
    ptable->resize(ptable, start + npage, &p);
    *out_ptr = p.data;
    TLB* pmap = &(mgr->pmap);
    pmap->set(pmap, *out_ptr, &p);
  }
  mgr->interface.vleak_size++;
#if TVM_CRT_DEBUG > 1
  TVMLogf("allocate: addr=%p, start=%" PRId64 "/%zu, npage=%" PRId64 ", vleak=%d\n", data, start,
          ptable->max_pages, npage, mgr->interface.vleak_size);
#endif  // TVM_CRT_DEBUG
  return kTvmErrorNoError;
}

/*!
 * \brief Reallocate memory from manager
 * \param ptr Pointer holding a pointer to the memory area to be reallocated
 * \param num_bytes The size of memory now required.
 * \return kTvmErrorNoError on success.
 */
tvm_crt_error_t PageMemoryManager_Realloc(MemoryManagerInterface* interface, void** ptr,
                                          tvm_index_t num_bytes) {
  MemoryManager* mgr = (MemoryManager*)interface;

  uint8_t* data = *((uint8_t**)ptr);  // NOLINT(*)
  PageTable* ptable = &(mgr->ptable);
  TLB* pmap = &(mgr->pmap);
  MultiMap* free_map = &(mgr->free_map);
  tvm_index_t start = 0;
  tvm_index_t npage = (num_bytes + ptable->page_size_bytes - 1) / ptable->page_size_bytes;
  if (ptr) {
    // get page size for given pointer
    CHECK_NE(pmap->num_pages, 0, "invalid translation look-aside buffer.");
    PageEntry* entry = pmap->find(pmap, (uint8_t*)ptr);  // NOLINT(*)
    CHECK_NE(entry, 0, "no valid page entry found.");
    Page* pptr = &(entry->page);
    // if the page size is smaller than target page size,
    // try allocate new space
    if (pptr->num_pages < npage) {
      // TODO(liangfu): found out whether we can extend current entry
      //
      // insert new page entry
      IndexedEntry* it = free_map->lower_bound(free_map, npage);
      if (it != free_map->end(free_map)) {
        data = it->page.data;
        start = it->page.ptable_begin;
        npage = it->page.num_pages;
        free_map->erase(free_map, it);
      } else {
        start = ptable->num_pages;
        if ((unsigned)(start + npage) > ptable->max_pages) {
#if TVM_CRT_DEBUG > 1
          TVMLogf("insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "",
                  start, npage, start + npage);
#endif
          return kTvmErrorPlatformNoMemory;
        }
        Page p = PageCreate(mgr->ptable.memory_pool, mgr->ptable.page_size_bytes, start, npage);
        ptable->resize(ptable, start + npage, &p);
        data = p.data;
        pmap->set(pmap, data, &p);
      }
      // copy previous data to the new entry
      memcpy(data, ptr, ptable->page_size_bytes * pptr->num_pages);
      // release memory
      free_map->insert(free_map, pptr->num_pages, pptr);
    } else {
      start = pptr->ptable_begin;
    }
  } else {
    IndexedEntry* it = free_map->lower_bound(free_map, npage);
    if (it != free_map->end(free_map)) {
      Page p = it->page;
      free_map->erase(free_map, it);
      data = p.data;
      start = p.ptable_begin;
      npage = p.num_pages;
    } else {
      PageTable* ptable = &(mgr->ptable);
      start = ptable->num_pages;
      if ((unsigned)(start + npage) > ptable->max_pages) {
#if TVM_CRT_DEBUG > 1
        TVMLogf("insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "",
                start, npage, start + npage);
#endif
        /* insert page entry */
        Page p = PageCreate(mgr->ptable.memory_pool, mgr->ptable.page_size_bytes, start, npage);
        ptable->resize(ptable, start + npage, &p);
        data = p.data;
        TLB* pmap = &(mgr->pmap);
        pmap->set(pmap, data, &p);
      }
      mgr->interface.vleak_size++;
    }
  }
#if TVM_CRT_DEBUG > 1
  TVMLogf("reallocate: addr=%p, start=%" PRId64 "/%zu, npage=%" PRId64 ", vleak=%d, size=%zu", data,
          start, mgr->ptable.max_pages, npage, mgr->interface.vleak_size, size);
#endif  // TVM_CRT_DEBUG
  return kTvmErrorNoError;
}

/*!
 * \brief Free the memory.
 * \param interface Pointer to this structure.
 * \param ptr A pointer returned from TVMPlatformMemoryAllocate which should be free'd.
 * \param dev Execution device passed to TVMPlatformMemoryAllocate. Fixed to {kDLCPU, 0}.
 * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
 */
tvm_crt_error_t PageMemoryManager_Free(MemoryManagerInterface* interface, void* ptr, DLDevice dev) {
  MemoryManager* mgr = (MemoryManager*)interface;

  TLB* pmap = &(mgr->pmap);
  CHECK_NE(pmap->num_pages, 0, "invalid translation look-aside buffer.");
  PageEntry* entry = pmap->find(pmap, (uint8_t*)ptr);  // NOLINT(*)
  CHECK_NE(entry, 0, "no valid page entry found.");
  Page* p = &(entry->page);
  MultiMap* free_map = &(mgr->free_map);
  free_map->insert(free_map, p->num_pages, p);
  mgr->interface.vleak_size--;
#if TVM_CRT_DEBUG > 1
  TVMLogf("release: addr=%p, start=%" PRId64 "/%zu, npage=%zu, vleak=%d", ptr,
          entry->page.ptable_begin, mgr->ptable.max_pages, entry->page.num_pages,
          mgr->interface.vleak_size);
#endif  // TVM_CRT_DEBUG
  return kTvmErrorNoError;
}

tvm_crt_error_t PageMemoryManagerCreate(MemoryManagerInterface** interface, uint8_t* memory_pool,
                                        size_t memory_pool_size_bytes,
                                        size_t page_size_bytes_log2) {
  memset(memory_pool, 0, memory_pool_size_bytes);

  // Allocate enough space for MAX_PAGES.
  size_t page_size_bytes = 1 << page_size_bytes_log2;
  size_t metadata_bytes_per_page = sizeof(Page) + sizeof(PageEntry) + sizeof(IndexedEntry);
  size_t bytes_needed_per_page = page_size_bytes + metadata_bytes_per_page;
  size_t num_pages = (memory_pool_size_bytes - sizeof(MemoryManager)) / bytes_needed_per_page;

  uint8_t* metadata_cursor = memory_pool + (num_pages << page_size_bytes_log2);
  MemoryManager* manager = (MemoryManager*)metadata_cursor;
  *interface = &manager->interface;
  /* handle MemoryManager member functions */
  manager->interface.Allocate = PageMemoryManager_Allocate;
  //  manager->Realloc = MemoryManager_Reallocate;
  manager->interface.Free = PageMemoryManager_Free;

  metadata_cursor += sizeof(MemoryManager);

  manager->interface.Allocate = PageMemoryManager_Allocate;
  manager->interface.Free = PageMemoryManager_Free;
  manager->ptable.memory_pool = memory_pool;

  /* handle PageTable member functions */
  manager->ptable.page = (Page*)metadata_cursor;
  metadata_cursor += sizeof(Page) * num_pages;

  manager->ptable.page_size_bytes = (1 << page_size_bytes_log2);
  manager->ptable.max_pages = num_pages;
  manager->ptable.resize = PageTable_Resize;

  /* handle TLB member functions */
  manager->pmap.entries = (PageEntry*)metadata_cursor;
  metadata_cursor += sizeof(PageEntry) * num_pages;
  manager->pmap.max_pages = num_pages;
  manager->pmap.num_pages = 0;

  manager->pmap.set = TLB_Set;
  manager->pmap.find = TLB_Find;
  /* handle free_map member functions */
  manager->free_map.entries = (IndexedEntry*)metadata_cursor;
  metadata_cursor += sizeof(IndexedEntry) * num_pages;
  manager->free_map.max_entries = num_pages;
  manager->free_map.lower_bound = MultiMap_LowerBound;
  manager->free_map.end = MultiMap_End;
  manager->free_map.erase = MultiMap_Erase;
  manager->free_map.insert = MultiMap_Insert;

  return kTvmErrorNoError;
}
