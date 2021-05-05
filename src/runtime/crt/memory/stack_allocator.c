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
#ifdef TVM_CRT_STACK_ALLOCATOR_ENABLE_FIFO_CHECK
#include <tvm/runtime/crt/logging.h>
#endif

tvm_crt_error_t StackMemoryManager_Allocate(tvm_workspace_t* tvm_runtime_workspace, int32_t nbytes,
                                            void** current_alloc) {
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
#ifdef TVM_CRT_STACK_ALLOCATOR_ENABLE_FIFO_CHECK
  if (next_alloc + STACK_ALLOCATOR_TAG_SIZE_BYTES > workspace_end) {
    return kTvmErrorPlatformNoMemory;
  }
  const uint32_t total_size = (nbytes + offset_bytes + STACK_ALLOCATOR_TAG_SIZE_BYTES);
  *((uint32_t*)next_alloc) = total_size ^ STACK_ALLOCATOR_TAG;
  next_alloc += STACK_ALLOCATOR_TAG_SIZE_BYTES;
#endif

  tvm_runtime_workspace->next_alloc = next_alloc;
  return kTvmErrorNoError;
}

tvm_crt_error_t StackMemoryManager_Free(tvm_workspace_t* tvm_runtime_workspace, void* ptr) {
#ifdef TVM_CRT_STACK_ALLOCATOR_ENABLE_FIFO_CHECK
  uint32_t tag = *(((uint32_t*)tvm_runtime_workspace->next_alloc) - 1);
  uint32_t actual_size = (tvm_runtime_workspace->next_alloc - (uint8_t*)ptr);
  uint32_t expected_size = tag ^ STACK_ALLOCATOR_TAG;
  CHECK_EQ(expected_size, actual_size, "Deallocation not in FIFO ordering");
#endif
  tvm_runtime_workspace->next_alloc = ptr;
  return kTvmErrorNoError;
}

tvm_crt_error_t StackMemoryManager_Init(tvm_workspace_t* tvm_runtime_workspace,
                                        uint8_t* g_aot_memory, size_t workspace_size) {
  tvm_runtime_workspace->next_alloc = g_aot_memory;
  tvm_runtime_workspace->workspace = g_aot_memory;
  tvm_runtime_workspace->workspace_size = workspace_size;
  return kTvmErrorNoError;
}
