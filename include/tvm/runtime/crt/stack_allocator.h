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

tvm_crt_error_t StackMemoryManager_Init(tvm_workspace_t* tvm_runtime_workspace,
                                        uint8_t* g_aot_memory, size_t workspace_size);

tvm_crt_error_t StackMemoryManager_Allocate(tvm_workspace_t* tvm_runtime_workspace, int32_t nbytes,
                                            void**);

tvm_crt_error_t StackMemoryManager_Free(tvm_workspace_t* tvm_runtime_workspace, void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_STACK_ALLOCATOR_H_
