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
#include <tvm/runtime/crt/stack_allocator.h>

tvm_crt_error_t StackMemoryManager_Allocate_Body(tvm_workspace_t* tvm_runtime_workspace,
                                                 int32_t nbytes, void** current_alloc,
                                                 uint8_t do_lifo_check) {
  // reserve bytes at the end of the allocation such that
  // next_alloc % TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES == 0.
  uint32_t offset_bytes =
      (TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES - nbytes) & (TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES - 1);
  uint8_t* workspace_end = tvm_runtime_workspace->workspace + tvm_runtime_workspace->workspace_size;
  if (tvm_runtime_workspace->next_alloc + nbytes + offset_bytes > workspace_end) {
    return kTvmErrorPlatformNoMemory;
  }
  (*current_alloc) = tvm_runtime_workspace->next_alloc;
  uint8_t* next_alloc = tvm_runtime_workspace->next_alloc + nbytes + offset_bytes;
  if (do_lifo_check != 0) {
    if (next_alloc + STACK_ALLOCATOR_TAG_SIZE_BYTES > workspace_end) {
      return kTvmErrorPlatformNoMemory;
    }
    const uint32_t total_size = (nbytes + offset_bytes + STACK_ALLOCATOR_TAG_SIZE_BYTES);
    *((uint32_t*)next_alloc) = total_size ^ STACK_ALLOCATOR_TAG;
    next_alloc += STACK_ALLOCATOR_TAG_SIZE_BYTES;
  }

  tvm_runtime_workspace->next_alloc = next_alloc;
  return kTvmErrorNoError;
}

tvm_crt_error_t StackMemoryManager_Allocate(tvm_workspace_t* tvm_runtime_workspace, int32_t nbytes,
                                            void** current_alloc) {
  uint8_t do_lifo_check = 0;
#ifdef TVM_CRT_STACK_ALLOCATOR_ENABLE_LIFO_CHECK
  do_lifo_check = 1;
#endif
  return StackMemoryManager_Allocate_Body(tvm_runtime_workspace, nbytes, current_alloc,
                                          do_lifo_check);
}

tvm_crt_error_t StackMemoryManager_Free_Body(tvm_workspace_t* tvm_runtime_workspace, void* ptr,
                                             uint8_t do_lifo_check) {
  if (do_lifo_check != 0) {
    uint32_t tag = *(((uint32_t*)tvm_runtime_workspace->next_alloc) - 1);
    uint32_t actual_size = (tvm_runtime_workspace->next_alloc - (uint8_t*)ptr);
    uint32_t expected_size = tag ^ STACK_ALLOCATOR_TAG;
    if (expected_size != actual_size) {
      return kTvmErrorPlatformStackAllocBadFree;
    }
  }
  tvm_runtime_workspace->next_alloc = (uint8_t*)ptr;
  return kTvmErrorNoError;
}

tvm_crt_error_t StackMemoryManager_Free(tvm_workspace_t* tvm_runtime_workspace, void* ptr) {
  uint8_t do_lifo_check = 0;
#ifdef TVM_CRT_STACK_ALLOCATOR_ENABLE_LIFO_CHECK
  do_lifo_check = 1;
#endif
  return StackMemoryManager_Free_Body(tvm_runtime_workspace, ptr, do_lifo_check);
}

tvm_crt_error_t StackMemoryManager_Init(tvm_workspace_t* tvm_runtime_workspace,
                                        uint8_t* g_aot_memory, size_t workspace_size) {
  // We need to round up g_aot_memory in case it is not aligned to
  // TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES.
  uintptr_t unaligned_mask = TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES - 1;
  uint8_t* memory_aligned =
      (uint8_t*)(((uintptr_t)g_aot_memory + unaligned_mask) & ~unaligned_mask);
  uint32_t offset = (uintptr_t)(memory_aligned - g_aot_memory);

  tvm_runtime_workspace->next_alloc = memory_aligned;
  tvm_runtime_workspace->workspace = memory_aligned;
  tvm_runtime_workspace->workspace_size = workspace_size - offset;
  return kTvmErrorNoError;
}
