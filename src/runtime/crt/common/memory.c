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
 * \file memory.c
 * \brief Virtual memory manager
 *
 * To maximize portability, thread-safe feature has been dropped for now.
 */

#include <inttypes.h>

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/memory.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/internal/common/logging.h>

#include "crt_config.h"
#include "memory_internal.h"

/*! \brief A page in the DRAM */
typedef struct Page {
  /*! \brief Start location in page table */
  tvm_index_t ptable_begin;
  /*! \brief The total number of pages */
  tvm_index_t num_pages;
  /*! \brief Data */
  uint8_t* data;
} Page;

// construct a new page
Page PageCreate(uint8_t* memory_pool,
                size_t page_size_bytes,
                tvm_index_t ptable_begin,
                tvm_index_t num_pages) {
  Page page;
  page.ptable_begin = ptable_begin;
  page.num_pages = num_pages;
  page.data = memory_pool + ptable_begin * page_size_bytes;
  return page;
}

typedef struct PageTable {
  // Pointer to beginning of memory pool.
  uint8_t* memory_pool;
  // Size of one page.
  size_t page_size_bytes;

  Page* page;
  size_t max_pages;
  size_t num_pages;
  void (*resize)(struct PageTable* ptable, size_t size, Page* page);
} PageTable;

void PageTable_Resize(struct PageTable* ptable, size_t new_size, Page* page) {
  CHECK_LE(ptable->num_pages, new_size, "size value (%zu) is smaller than expected (%zu).", new_size,
           ptable->num_pages);
  for (uint32_t idx = ptable->num_pages; idx < new_size; idx++) {
    ptable->page[idx] = *page;
  }
  ptable->num_pages = new_size;
}

typedef struct PageEntry {
  uint8_t* addr;
  Page page;
} PageEntry;

typedef struct TLB {
  PageEntry* entries;
  size_t max_pages;
  uint32_t num_pages;
  void (*set)(struct TLB* tlb, uint8_t* data, Page* page);
  PageEntry* (*find)(struct TLB* tlb, uint8_t* data);
} TLB;

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

typedef struct IndexedEntry {
  tvm_index_t index;
  Page page;
} IndexedEntry;

typedef struct MultiMap {
  IndexedEntry* entries;
  size_t max_entries;
  size_t num_entries;
  IndexedEntry* (*lower_bound)(struct MultiMap* map, uint32_t npage);
  IndexedEntry* (*end)(struct MultiMap* map);
  void (*erase)(struct MultiMap* map, IndexedEntry* entry);
  void (*insert)(struct MultiMap* map, uint32_t npage, Page* p);
} MultiMap;

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
      memcpy(map->entries + idx, map->entries + (idx + 1),
             sizeof(IndexedEntry) * (map->num_entries - idx));
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
 * \brief DRAM memory manager
 *  Implements simple paging to allow physical address translation.
 */
typedef struct MemoryManager {
  /*!
   * \brief Allocate memory from manager
   * \param size The size of memory
   * \return The virtual address
   */
  void* (*Alloc)(struct MemoryManager* mgr, tvm_index_t size);
  /*!
   * \brief Allocate memory from manager
   * \param ptr The pointer to the memory area to be reallocated
   * \param size The size of memory
   * \return The virtual address
   */
  void* (*Realloc)(struct MemoryManager* mgr, void* ptr, tvm_index_t size);
  /*!
   * \brief Free the memory.
   * \param ptr The pointer to the memory to deallocate
   * \return The virtual address
   */
  void (*Free)(struct MemoryManager* mgr, void* data);

  // Physical address -> page
  PageTable ptable;
  // Virtual address -> page
  TLB pmap;
  // Free map
  MultiMap free_map;
} MemoryManager;

/*!
 * \brief Allocate memory from manager
 * \param size The size of memory
 * \return The virtual address
 */
void* MemoryManager_Alloc(MemoryManager* mgr, tvm_index_t size) {
  uint8_t* data = 0;
  PageTable* ptable = &(mgr->ptable);
  tvm_index_t npage = (size + ptable->page_size_bytes - 1) / ptable->page_size_bytes;
  MultiMap* free_map = &(mgr->free_map);
  IndexedEntry* it = free_map->lower_bound(free_map, npage);
  tvm_index_t start = 0;
  if (it != free_map->end(free_map)) {
    Page p = it->page;
    free_map->erase(free_map, it);
    data = p.data;
    start = p.ptable_begin;
    npage = p.num_pages;
  } else {
    start = ptable->num_pages;
    CHECK_LE((unsigned)(start + npage), ptable->max_pages,
             "insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "", start,
             npage, start + npage);
    /* insert page entry */
    Page p = PageCreate(ptable->memory_pool, ptable->page_size_bytes, start, npage);
    ptable->resize(ptable, start + npage, &p);
    data = p.data;
    TLB* pmap = &(mgr->pmap);
    pmap->set(pmap, data, &p);
  }
  vleak_size++;
#if TVM_CRT_DEBUG > 1
  printf("allocate: addr=%p, start=%d/%d, npage=%d, vleak=%d\n", data, start, ptable->max_pages,
         npage, vleak_size);
#endif  // TVM_CRT_DEBUG
  return data;
}

/*!
 * \brief Reallocate memory from manager
 * \param ptr The pointer to the memory area to be reallocated
 * \param size The size of memory
 * \return The virtual address
 */
void* MemoryManager_Realloc(MemoryManager* mgr, void* ptr, tvm_index_t size) {
  uint8_t* data = (uint8_t*)ptr;  // NOLINT(*)
  PageTable* ptable = &(mgr->ptable);
  TLB* pmap = &(mgr->pmap);
  MultiMap* free_map = &(mgr->free_map);
  tvm_index_t start = 0;
  tvm_index_t npage = (size + ptable->page_size_bytes - 1) / ptable->page_size_bytes;
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
        CHECK_LE((unsigned)(start + npage), ptable->max_pages,
                 "insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "",
                 start, npage, start + npage);
        Page p = PageCreate(mgr->ptable.memory_pool, mgr->ptable.num_pages, start, npage);
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
      CHECK_LE((unsigned)(start + npage), ptable->max_pages,
               "insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "",
               start, npage, start + npage);
      /* insert page entry */
      Page p = PageCreate(mgr->ptable.memory_pool, mgr->ptable.num_pages, start, npage);
      ptable->resize(ptable, start + npage, &p);
      data = p.data;
      TLB* pmap = &(mgr->pmap);
      pmap->set(pmap, data, &p);
    }
    vleak_size++;
  }
#if TVM_CRT_DEBUG > 1
  printf("reallocate: addr=%p, start=%d/%d, npage=%d, vleak=%d, size=%d\n", data, start,
         mgr->ptable.max_pages, npage, vleak_size, size);
#endif  // TVM_CRT_DEBUG
  return data;
}

/*!
 * \brief Free the memory.
 * \param ptr The pointer to the memory to deallocate
 * \return The virtual address
 */
void MemoryManager_Free(MemoryManager* mgr, void* ptr) {
  TLB* pmap = &(mgr->pmap);
  CHECK_NE(pmap->num_pages, 0, "invalid translation look-aside buffer.");
  PageEntry* entry = pmap->find(pmap, (uint8_t*)ptr);  // NOLINT(*)
  CHECK_NE(entry, 0, "no valid page entry found.");
  Page* p = &(entry->page);
  MultiMap* free_map = &(mgr->free_map);
  free_map->insert(free_map, p->num_pages, p);
  vleak_size--;
#if TVM_CRT_DEBUG > 1
  printf("release: addr=%p, start=%d/%d, npage=%d, vleak=%d\n", ptr, entry->page.ptable_begin,
         mgr->ptable.max_pages, entry->page.num_pages, vleak_size);
#endif  // TVM_CRT_DEBUG
}

static bool g_memory_manager_initialized = 0;
static MemoryManager g_memory_manager;

MemoryManager* MemoryManagerCreate(uint8_t* memory_pool,
                                   size_t memory_pool_size_bytes,
                                   size_t page_size_bytes_log2) {
  if (g_memory_manager_initialized) {
    TVMPlatformAbort(-1);
  }

  size_t num_pages = memory_pool_size_bytes / ((1 << page_size_bytes_log2) + sizeof(Page) + sizeof(PageEntry));

  memset(&g_memory_manager, 0, sizeof(MemoryManager));
  memset(memory_pool, 0, sizeof(memory_pool_size_bytes));

  /* handle MemoryManager member functions */
  g_memory_manager.Alloc = MemoryManager_Alloc;
  g_memory_manager.Realloc = MemoryManager_Realloc;
  g_memory_manager.Free = MemoryManager_Free;

  /* handle PageTable member functions */
  g_memory_manager.ptable.page = (Page*) memory_pool;

  // Allocate enough space for MAX_PAGES.
  size_t metadata_num_pages =
    ((sizeof(Page) + sizeof(PageEntry)) * num_pages + ((1 << page_size_bytes_log2) - 1)) >> page_size_bytes_log2;
  g_memory_manager.ptable.memory_pool = memory_pool + (metadata_num_pages << page_size_bytes_log2);

  g_memory_manager.ptable.page_size_bytes = (1 << page_size_bytes_log2);
  g_memory_manager.ptable.max_pages = num_pages;
  g_memory_manager.ptable.resize = PageTable_Resize;
  /* handle TLB member functions */
  g_memory_manager.pmap.entries = (PageEntry*) (memory_pool + (sizeof(Page) * num_pages));

  g_memory_manager.pmap.set = TLB_Set;
  g_memory_manager.pmap.find = TLB_Find;
  /* handle free_map member functions */
  g_memory_manager.free_map.max_entries = num_pages;
  g_memory_manager.free_map.lower_bound = MultiMap_LowerBound;
  g_memory_manager.free_map.end = MultiMap_End;
  g_memory_manager.free_map.erase = MultiMap_Erase;
  g_memory_manager.free_map.insert = MultiMap_Insert;

  g_memory_manager_initialized = true;
  return &g_memory_manager;
}

MemoryManager* TVMGetGlobalMemoryManager() {
  /* initialize once */
  if (!g_memory_manager_initialized) {
    TVMPlatformAbort(-1);
  }
  return &g_memory_manager;
}

/** \brief Allocate memory from manager */
void* vmalloc(size_t size) {
  MemoryManager* mgr = TVMGetGlobalMemoryManager();
  return mgr->Alloc(mgr, size);
}

/** \brief Reallocate memory from manager */
void* vrealloc(void* ptr, size_t size) {
  MemoryManager* mgr = TVMGetGlobalMemoryManager();
  return mgr->Realloc(mgr, ptr, size);
}

/** \brief Release memory from manager */
void vfree(void* ptr) {
  MemoryManager* mgr = TVMGetGlobalMemoryManager();
  mgr->Free(mgr, ptr);
}

int vleak_size = 0;
