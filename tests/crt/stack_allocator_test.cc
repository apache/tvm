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

#include "../../src/runtime/crt/memory/stack_allocator.c"

#include <gtest/gtest.h>
#include <tvm/runtime/crt/stack_allocator.h>

// Check with LIFO checks enabled for stack allocator
#define TVM_CRT_STACK_ALLOCATOR_ENABLE_LIFO_CHECK

// Number of memory misalignment in bytes
#define NUM_MEMORY_MISALIGNMENT_BYTES 1

/*!
 * Align memory pointer.
 * This function modifies memory_ptr to adjust alignment.
 * \return Number of memory offset.
 */
static uint32_t align_pointer(uint8_t** memory_ptr) {
  uint32_t extra = (uintptr_t)(*memory_ptr) % TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES;
  uint32_t offset =
      (TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES - extra) & (TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES - 1);
  *memory_ptr += offset;
  return offset;
}

/*!
 * Add misalignment to memory pointer.
 * This function modifies memory_ptr.
 * \return Number of memory offset.
 */
static uint32_t misalign_pointer(uint8_t** memory_ptr) {
  uint32_t extra = (uintptr_t)(*memory_ptr) % TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES;
  if (extra == 0) {
    *memory_ptr += NUM_MEMORY_MISALIGNMENT_BYTES;
    return 1;
  }
  return 0;
}

/*
 * Tests allocations are properly aligned when allocated.
 */
TEST(StackAllocatorTest, Allocate) {
  static uint8_t model_memory[128];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = align_pointer(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr,
                                    sizeof(model_memory) - offset),
            kTvmErrorNoError);

  void* block_one = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_one, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_one, &model_memory_ptr[0]);

  void* block_two = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 2, &block_two, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_two, &model_memory_ptr[16 + STACK_ALLOCATOR_TAG_SIZE_BYTES]);

  void* two_blocks = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 24, &two_blocks, 1),
            kTvmErrorNoError);
  ASSERT_EQ(two_blocks, &model_memory_ptr[32 + 2 * STACK_ALLOCATOR_TAG_SIZE_BYTES]);

  void* block_three = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_three, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_three, &model_memory_ptr[64 + 3 * STACK_ALLOCATOR_TAG_SIZE_BYTES]);
}

/*
 * Tests resetting the stack after dealloc.
 */
TEST(StackAllocatorTest, Free) {
  static uint8_t model_memory[80];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = align_pointer(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr,
                                    sizeof(model_memory) - offset),
            kTvmErrorNoError);

  void* block_one = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_one, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_one, &model_memory_ptr[0]);

  void* block_two = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_two, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_two, &model_memory_ptr[16 + STACK_ALLOCATOR_TAG_SIZE_BYTES]);
  ASSERT_EQ(kTvmErrorNoError, StackMemoryManager_Free_Body(&tvm_runtime_workspace, block_two, 1));

  void* two_blocks = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 2, &two_blocks, 1),
            kTvmErrorNoError);
  ASSERT_EQ(two_blocks, &model_memory_ptr[16 + STACK_ALLOCATOR_TAG_SIZE_BYTES]);
  ASSERT_EQ(kTvmErrorNoError, StackMemoryManager_Free_Body(&tvm_runtime_workspace, two_blocks, 1));

  void* block_three = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_three, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_three, &model_memory_ptr[16 + STACK_ALLOCATOR_TAG_SIZE_BYTES]);
}

/*
 * Tests we return NULL if we over allocate.
 */
TEST(StackAllocatorTest, OverAllocate) {
  static uint8_t model_memory[80];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = align_pointer(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr,
                                    sizeof(model_memory) - offset),
            kTvmErrorNoError);

  void* block_one = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_one, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_one, &model_memory_ptr[0]);

  void* block_two = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_two, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_two, &model_memory_ptr[16 + STACK_ALLOCATOR_TAG_SIZE_BYTES]);

  void* two_blocks = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 64, &two_blocks, 1),
            kTvmErrorPlatformNoMemory);
  ASSERT_EQ(two_blocks, (void*)NULL);
}

/*
 * Test for out-of-order memory deallocation.
 */
TEST(StackAllocatorTest, FreeOutOfOrder) {
  static uint8_t model_memory[80];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = align_pointer(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr,
                                    sizeof(model_memory) - offset),
            kTvmErrorNoError);

  void* block_one = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_one, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_one, &model_memory_ptr[0]);

  void* block_two = NULL;
  ASSERT_EQ(StackMemoryManager_Allocate_Body(&tvm_runtime_workspace, 1, &block_two, 1),
            kTvmErrorNoError);
  ASSERT_EQ(block_two, &model_memory_ptr[16 + STACK_ALLOCATOR_TAG_SIZE_BYTES]);

  ASSERT_EQ(StackMemoryManager_Free_Body(&tvm_runtime_workspace, block_one, 1),
            kTvmErrorPlatformStackAllocBadFree);
}

/*
 * Test for initial memory misalignment.
 */
TEST(StackAllocatorTest, InitialMemoryMisAlignment) {
  static uint8_t model_memory[80];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;

  // Add misaslignment to memory pointer
  uint32_t offset = misalign_pointer(&model_memory_ptr);

  // Calculate expected offset
  uint8_t* misaligned_ptr = model_memory_ptr;
  uint32_t alignment_offset = align_pointer(&misaligned_ptr);

  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr,
                                    sizeof(model_memory) - offset),
            kTvmErrorNoError);

  ASSERT_EQ(tvm_runtime_workspace.next_alloc, &model_memory_ptr[alignment_offset]);
  ASSERT_EQ(tvm_runtime_workspace.workspace_size, sizeof(model_memory) - offset - alignment_offset);
}
