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
#ifndef TVM_RUNTIME_CRT_STACK_ALLOCATOR_H_
#define TVM_RUNTIME_CRT_STACK_ALLOCATOR_H_
#include <stddef.h>
#include <stdint.h>

#include "crt_config.h"
#include "error_codes.h"

#define STACK_ALLOCATOR_TAG 0xabcd1234
#define STACK_ALLOCATOR_TAG_SIZE_BYTES 4

/*! Memory alignment for allocator */

#ifndef TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES
#define TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES 16
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint8_t* next_alloc;    // Pointer to the next block of TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES
  uint8_t* workspace;     // Pointer to start of the workspace
  size_t workspace_size;  // Total number of bytes in the workspace
} tvm_workspace_t;

/*!
 * \brief Initialize the stack-based memory manager
 *
 * \param tvm_runtime_workspace The tvm_workspace_t struct containing state
 * \param g_aot_memory The memory buffer used to allocate within
 * \param workspace_size The total size of the workspace buffer workspace
 */
tvm_crt_error_t StackMemoryManager_Init(tvm_workspace_t* tvm_runtime_workspace,
                                        uint8_t* g_aot_memory, size_t workspace_size);

/*!
 * \brief The intended user-facing function to allocate within the buffer. It wraps
 * StackMemoryManager_Allocate_Body enable and disable the LIFO check that is useful for debugging
 * the AoT codegen.
 *
 * \param tvm_runtime_workspace The tvm_workspace_t struct containing state
 * \param nbytes The number of bytes required for the allocation
 * \param current_alloc The pointer-to-pointer to be populated with the allocated address
 */
tvm_crt_error_t StackMemoryManager_Allocate(tvm_workspace_t* tvm_runtime_workspace, int32_t nbytes,
                                            void** current_alloc);

/*!
 * \brief The internal function that accepts allocate inputs and an extra byte to say to enable the
 * LIFO check that is useful in debugging for debugging the AoT codegen.
 *
 * \param tvm_runtime_workspace The tvm_workspace_t struct containing state
 * \param nbytes The number of bytes required for the allocation
 * \param current_alloc The pointer-to-pointer to be populated with the allocated address
 * \param do_lifo_check This being non-zero indicates to perform a check LIFO pattern Allocs/Frees
 */
tvm_crt_error_t StackMemoryManager_Allocate_Body(tvm_workspace_t* tvm_runtime_workspace,
                                                 int32_t nbytes, void** current_alloc,
                                                 uint8_t do_lifo_check);

/*!
 * \brief The intended user-facing function to free the tensor within the buffer. It wraps
 * StackMemoryManager_Free_Body enable and disable the stack allocator
 *
 * \param tvm_runtime_workspace The tvm_workspace_t struct containing state
 * \param ptr The base pointer of the tensor to be free'd
 */
tvm_crt_error_t StackMemoryManager_Free(tvm_workspace_t* tvm_runtime_workspace, void* ptr);

/*!
 * \brief The internal function that accepts free inputs and an extra byte to say to enable the LIFO
 * check that is useful in debugging for debugging the AoT codegen.
 *
 * \param tvm_runtime_workspace The tvm_workspace_t struct containing state
 * \param ptr The base pointer of tensor to be free'd within the workspace buffer
 * \param do_lifo_check This being non-zero indicates to perform a check LIFO pattern Allocs/Frees
 */
tvm_crt_error_t StackMemoryManager_Free_Body(tvm_workspace_t* tvm_runtime_workspace, void* ptr,
                                             uint8_t do_lifo_check);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_STACK_ALLOCATOR_H_
