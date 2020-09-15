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
 * \file runtime/crt/include/tvm/runtime/crt/internal/common/memory.h
 * \brief Defines data types and functions used in the internal memory manager.
 *     Exposed for testing.
 */

#ifndef TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_COMMON_MEMORY_H_
#define TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_COMMON_MEMORY_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>

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
 * Exposed for testing.
 *
 * \param manager The memory manager to initialize.
 * \param memory_pool Pointer to the global memory pool used by the CRT.
 * \param memory_pool_size_bytes Size of `memory_pool`, in bytes.
 * \param page_size_bytes_log2 log2 of the page size, in bytes.
 */
void MemoryManagerCreate(MemoryManager* manager, uint8_t* memory_pool,
                         size_t memory_pool_size_bytes, size_t page_size_bytes_log2);

/*!
 * Initialize the global memory manager.
 *
 * Call this function once before invoking any other CRT functions beginning with `TVM`.
 * Repeated calls will cause TVMPlatformAbort to be invoked.
 * \param memory_pool Pointer to the global memory pool used by the CRT.
 * \param memory_pool_size_bytes Size of `memory_pool`, in bytes.
 * \param page_size_bytes_log2 log2 of the page size, in bytes.
 * \return An error code indicating the status of the operation.
 */
tvm_crt_error_t TVMInitializeGlobalMemoryManager(uint8_t* memory_pool,
                                                 size_t memory_pool_size_bytes,
                                                 size_t page_size_bytes_log2);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_INCLUDE_TVM_RUNTIME_CRT_INTERNAL_COMMON_MEMORY_H_
