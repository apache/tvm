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
 * \file runtime/crt/include/tvm/runtime/crt/internal/memory/page_allocator.h
 * \brief Defines data types and functions used in the internal memory manager.
 *     Exposed for testing.
 */

#ifndef TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_MEMORY_PAGE_ALLOCATOR_H_
#define TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_MEMORY_PAGE_ALLOCATOR_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/page_allocator.h>

#include "crt_config.h"

#ifdef __cplusplus
extern "C" {
#endif

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
Page PageCreate(uint8_t* memory_pool, size_t page_size_bytes, tvm_index_t ptable_begin,
                tvm_index_t num_pages);

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

/*!
 * \brief DRAM memory manager
 *  Implements simple paging to allow physical address translation.
 */
typedef struct MemoryManager {
  // Public interface for this object.
  MemoryManagerInterface interface;
  // Physical address -> page
  PageTable ptable;
  // Virtual address -> page
  TLB pmap;
  // Free map
  MultiMap free_map;
} MemoryManager;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_MEMORY_PAGE_ALLOCATOR_H_
