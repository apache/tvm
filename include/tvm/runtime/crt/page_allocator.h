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
 * \file tvm/runtime/crt/page_allocator.h
 * \brief An implementation of a dynamic memory allocator for microcontrollers.
 */

#ifndef TVM_RUNTIME_CRT_PAGE_ALLOCATOR_H_
#define TVM_RUNTIME_CRT_PAGE_ALLOCATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>

extern int vleak_size;

typedef struct MemoryManagerInterface MemoryManagerInterface;

struct MemoryManagerInterface {
  /*!
   * \brief Allocate a chunk of memory.
   * \param interface Pointer to this structure.
   * \param num_bytes Number of bytes requested.
   * \param dev Execution device that will be used with the allocated memory. Must be {kDLCPU, 0}.
   * \param out_ptr A pointer to which is written a pointer to the newly-allocated memory.
   * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
   */
  tvm_crt_error_t (*Allocate)(MemoryManagerInterface* interface, size_t num_bytes, DLDevice dev,
                              void** out_ptr);

  /*!
   * \brief Free a chunk of previously-used memory.
   *
   * \param interface Pointer to this structure.
   * \param ptr A pointer returned from TVMPlatformMemoryAllocate which should be free'd.
   * \param dev Execution device passed to TVMPlatformMemoryAllocate. Fixed to {kDLCPU, 0}.
   * \return kTvmErrorNoError if successful; a descriptive error code otherwise.
   */
  tvm_crt_error_t (*Free)(MemoryManagerInterface* interface, void* ptr, DLDevice dev);

  /*! \brief Used in testing; the number of allocated objects. */
  int vleak_size;
};

/*!
 * Exposed for testing.
 *
 * \param manager Pointer, initialized with the new MemoryManager.
 * \param memory_pool Pointer to the global memory pool used by the CRT.
 * \param memory_pool_size_bytes Size of `memory_pool`, in bytes.
 * \param page_size_bytes_log2 log2 of the page size, in bytes.
 * \return kTvmErrorNoError on success.
 */
tvm_crt_error_t PageMemoryManagerCreate(MemoryManagerInterface** manager, uint8_t* memory_pool,
                                        size_t memory_pool_size_bytes, size_t page_size_bytes_log2);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_PAGE_ALLOCATOR_H_
