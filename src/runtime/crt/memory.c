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
 * \brief Virtal memory manager
 *
 * To maximize portability, thread-safe feature has been dropped for now.
 */

#include <inttypes.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/memory.h>

#include "logging.h"

/*! Number of bits in a page */
#define TVM_CRT_PAGE_BITS (TVM_CRT_PAGE_BYTES << 3)

/*! \brief Translate log memory size into bytes */
#define TVM_CRT_VIRT_MEM_SIZE (1 << TVM_CRT_LOG_VIRT_MEM_SIZE)

/*! \brief Number of possible page entries in total */
#define TVM_CRT_MAX_PAGES (TVM_CRT_VIRT_MEM_SIZE / TVM_CRT_PAGE_BYTES)

/*! \brief Physical address type */
typedef uint32_t tvm_phy_addr_t;

/*! \brief The bits in page table */
static const tvm_phy_addr_t kPageBits = TVM_CRT_PAGE_BITS;

/*! \brief Page size, also the maximum allocable size */
static const tvm_phy_addr_t kPageSize = TVM_CRT_PAGE_BYTES;

/**
 * \brief Memory pool for virtual dynamic memory allocation
 */
static char g_memory_pool[TVM_CRT_VIRT_MEM_SIZE];

/*! \brief A page in the DRAM */
typedef struct Page {
  /*! \brief Start location in page table */
  tvm_index_t ptable_begin;
  /*! \brief The total number of pages */
  tvm_index_t num_pages;
  /*! \brief Data */
  char* data;
} Page;

// construct a new page
Page PageCreate(tvm_index_t ptable_begin, tvm_index_t num_pages) {
  Page page;
  page.ptable_begin = ptable_begin;
  page.num_pages = num_pages;
  page.data = g_memory_pool + ptable_begin * kPageSize;
  return page;
}

typedef struct PageTable {
  Page page[TVM_CRT_MAX_PAGES];
  uint32_t count;
  void (*resize)(struct PageTable* ptable, uint32_t size, Page* page);
} PageTable;

void PageTable_Resize(struct PageTable* ptable, uint32_t new_size, Page* page) {
  CHECK_LE(ptable->count, new_size, "size value (%d) is smaller than expected (%d).", new_size,
           ptable->count);
  for (uint32_t idx = ptable->count; idx < new_size; idx++) {
    ptable->page[idx] = *page;
  }
  ptable->count = new_size;
}

typedef struct PageEntry {
  char* addr;
  Page page;
} PageEntry;

typedef struct TLB {
  PageEntry entries[TVM_CRT_MAX_PAGES];
  uint32_t count;
  void (*set)(struct TLB* tlb, char* data, Page* page);
  PageEntry* (*find)(struct TLB* tlb, char* data);
} TLB;

void TLB_Set(TLB* tlb, char* data, Page* page) {
  PageEntry* entry = tlb->find(tlb, data);
  if (entry == 0) {
    tlb->entries[tlb->count].addr = data;
    tlb->entries[tlb->count].page = *page;
    tlb->count++;
  } else {
    entry->addr = data;
    entry->page = *page;
  }
}

PageEntry* TLB_Find(TLB* tlb, char* data) {
  PageEntry* entry = 0;
  for (uint32_t idx = 0; idx < tlb->count; idx++) {
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
  IndexedEntry entries[TVM_CRT_MAX_PAGES];
  uint32_t count;
  IndexedEntry* (*lower_bound)(struct MultiMap* map, uint32_t npage);
  IndexedEntry* (*end)(struct MultiMap* map);
  void (*erase)(struct MultiMap* map, IndexedEntry* entry);
  void (*insert)(struct MultiMap* map, uint32_t npage, Page* p);
} MultiMap;

IndexedEntry* MultiMap_LowerBound(struct MultiMap* map, uint32_t npage) {
  IndexedEntry* entry = 0;
  for (uint32_t idx = 0; idx < map->count; idx++) {
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
  for (uint32_t idx = 0; idx < map->count; idx++) {
    if ((map->entries + idx) == entry) {
      memcpy(map->entries + idx, map->entries + (idx + 1),
             sizeof(IndexedEntry) * (map->count - idx));
      map->count--;
      break;
    }
  }
}

void MultiMap_Insert(struct MultiMap* map, uint32_t npage, Page* p) {
  CHECK_LE(map->count + 1, TVM_CRT_MAX_PAGES, "invalid number of free pages.");
  for (uint32_t idx = map->count; idx < (map->count + npage); idx++) {
    map->entries[map->count].index = npage;
    map->entries[map->count].page = *p;
  }
  map->count++;
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
  char* data = 0;
  tvm_index_t npage = (size + kPageSize - 1) / kPageSize;
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
    PageTable* ptable = &(mgr->ptable);
    start = ptable->count;
    CHECK_LE((unsigned)(start + npage), (sizeof(g_memory_pool) / kPageSize),
             "insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "", start,
             npage, start + npage);
    /* insert page entry */
    Page p = PageCreate(start, npage);
    ptable->resize(ptable, start + npage, &p);
    data = p.data;
    TLB* pmap = &(mgr->pmap);
    pmap->set(pmap, data, &p);
  }
  vleak_size++;
#if TVM_CRT_DEBUG > 1
  printf("allocate: addr=%p, start=%d/%d, npage=%d, vleak=%d\n", data, start, TVM_CRT_MAX_PAGES,
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
  char* data = (char*)ptr;  // NOLINT(*)
  PageTable* ptable = &(mgr->ptable);
  TLB* pmap = &(mgr->pmap);
  MultiMap* free_map = &(mgr->free_map);
  tvm_index_t start = 0;
  tvm_index_t npage = (size + kPageSize - 1) / kPageSize;
  if (ptr) {
    // get page size for given pointer
    CHECK_NE(pmap->count, 0, "invalid translation look-aside buffer.");
    PageEntry* entry = pmap->find(pmap, (char*)ptr);  // NOLINT(*)
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
        start = ptable->count;
        CHECK_LE((unsigned)(start + npage), (sizeof(g_memory_pool) / kPageSize),
                 "insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "",
                 start, npage, start + npage);
        Page p = PageCreate(start, npage);
        ptable->resize(ptable, start + npage, &p);
        data = p.data;
        pmap->set(pmap, data, &p);
      }
      // copy previous data to the new entry
      memcpy(data, ptr, kPageSize * pptr->num_pages);
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
      start = ptable->count;
      CHECK_LE((unsigned)(start + npage), (sizeof(g_memory_pool) / kPageSize),
               "insufficient memory, start=%" PRId64 ", npage=%" PRId64 ", total=%" PRId64 "",
               start, npage, start + npage);
      /* insert page entry */
      Page p = PageCreate(start, npage);
      ptable->resize(ptable, start + npage, &p);
      data = p.data;
      TLB* pmap = &(mgr->pmap);
      pmap->set(pmap, data, &p);
    }
    vleak_size++;
  }
#if TVM_CRT_DEBUG > 1
  printf("reallocate: addr=%p, start=%d/%d, npage=%d, vleak=%d, size=%d\n", data, start,
         TVM_CRT_MAX_PAGES, npage, vleak_size, size);
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
  CHECK_NE(pmap->count, 0, "invalid translation look-aside buffer.");
  PageEntry* entry = pmap->find(pmap, (char*)ptr);  // NOLINT(*)
  CHECK_NE(entry, 0, "no valid page entry found.");
  Page* p = &(entry->page);
  MultiMap* free_map = &(mgr->free_map);
  free_map->insert(free_map, p->num_pages, p);
  vleak_size--;
#if TVM_CRT_DEBUG > 1
  printf("release: addr=%p, start=%d/%d, npage=%d, vleak=%d\n", ptr, entry->page.ptable_begin,
         TVM_CRT_MAX_PAGES, entry->page.num_pages, vleak_size);
#endif  // TVM_CRT_DEBUG
}

MemoryManager* MemoryManagerCreate() {
  static MemoryManager mgr;
  memset(&mgr, 0, sizeof(MemoryManager));
  /* handle MemoryManager member functions */
  mgr.Alloc = MemoryManager_Alloc;
  mgr.Realloc = MemoryManager_Realloc;
  mgr.Free = MemoryManager_Free;
  /* handle PageTable member functions */
  mgr.ptable.resize = PageTable_Resize;
  /* handle TLB member functions */
  mgr.pmap.set = TLB_Set;
  mgr.pmap.find = TLB_Find;
  /* handle free_map member functions */
  mgr.free_map.lower_bound = MultiMap_LowerBound;
  mgr.free_map.end = MultiMap_End;
  mgr.free_map.erase = MultiMap_Erase;
  mgr.free_map.insert = MultiMap_Insert;
  return &mgr;
}

MemoryManager* TVMGetGlobalMemoryManager() {
  /* initialize once */
  static uint32_t initialized = 0;
  static MemoryManager* mgr;
  if (!initialized) {
    mgr = MemoryManagerCreate();
    memset(g_memory_pool, 0, sizeof(g_memory_pool));
    initialized = 1;
  }
  return mgr;
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
