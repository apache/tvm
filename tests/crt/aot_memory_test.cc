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

#include <gtest/gtest.h>
#include <tvm/runtime/crt/stack_allocator.h>

#include "../../src/runtime/crt/memory/stack_allocator.c"
#include "platform.cc"

// Check with LIFO checks enabled for stack allocator
#define TVM_CRT_STACK_ALLOCATOR_ENABLE_LIFO_CHECK

/*!
 * Align memory pointer.
 * \return Number of memory offset.
 */
uint32_t memory_align(uint8_t** memory_ptr) {
  uint32_t extra = (uintptr_t)(*memory_ptr) % TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES;
  uint32_t offset =
      (TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES - extra) & (TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES - 1);
  *memory_ptr += offset;
  return offset;
}

/*!
 * Add misalignment to memory pointer.
 * \return Number of memory offset.
 */
uint32_t memory_add_misalignment(uint8_t** memory_ptr) {
  uint32_t extra = (uintptr_t)(*memory_ptr) % TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES;
  if (extra != 0) {
    *memory_ptr += 1;
    return 1;
  }
  return 0;
}

/*
 * Tests allocations are properly aligned when allocated.
 */
TEST(AOTMemory, Allocate) {
  static uint8_t model_memory[128];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = memory_align(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr, 128 - offset),
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
TEST(AOTMemory, Free) {
  static uint8_t model_memory[80];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = memory_align(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr, 80 - offset),
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
TEST(AOTMemory, OverAllocate) {
  static uint8_t model_memory[72];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = memory_align(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr, 80 - offset),
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
TEST(AOTMemory, FreeOutOfOrder) {
  static uint8_t model_memory[80];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;
  uint32_t offset = memory_align(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr, 80 - offset),
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
TEST(AOTMemory, InitialMemoryMisAlignment) {
  static uint8_t model_memory[80];
  tvm_workspace_t tvm_runtime_workspace;
  uint8_t* model_memory_ptr = model_memory;

  uint32_t offset = memory_add_misalignment(&model_memory_ptr);
  ASSERT_EQ(StackMemoryManager_Init(&tvm_runtime_workspace, model_memory_ptr, 80 - offset),
            kTvmErrorNoError);

  uint32_t next_alloc_offset =
      (uintptr_t)tvm_runtime_workspace.next_alloc % TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES;
  uint32_t workspace_offset =
      (uintptr_t)tvm_runtime_workspace.workspace % TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES;
  ASSERT_EQ(next_alloc_offset, 0);
  ASSERT_EQ(workspace_offset, 0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
