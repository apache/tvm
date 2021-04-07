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
#ifndef TVM_RUNTIME_CRT_STACK_MEMORY_H_
#define TVM_RUNTIME_CRT_STACK_MEMORY_H_
#include <stddef.h>
#include <stdint.h>

#include "error_codes.h"

/*! Memory alignment for allocator */

#ifndef TVM_RUNTIME_ALLOC_ALIGNMENT
#define TVM_RUNTIME_ALLOC_ALIGNMENT 16
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint8_t* next_alloc;   /** Pointer to the next block of bytes to allocate */
  uint8_t* workspace;    /** Pointer to start of the workspace */
  size_t workspace_size; /** Total number of bytes in the workspace */
} tvm_workspace_t;

void MemoryManager_Init(tvm_workspace_t* tvm_runtime_workspace, uint8_t* g_aot_memory,
                        size_t workspace_size);

void* MemoryManager_Allocate(tvm_workspace_t* tvm_runtime_workspace, int32_t nbytes);

tvm_crt_error_t MemoryManager_Free(tvm_workspace_t* tvm_runtime_workspace, void* ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_CRT_STACK_MEMORY_H_
