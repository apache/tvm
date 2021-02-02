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

#include "tvm_backend.h"

// TODO(Mousius) - Move memory allocation to individual networks
extern tvm_workspace_t* tvm_runtime_workspace;

/*
 * Tests allocations are properly aligned when allocated
 */
TEST(AOTMemory, Allocate) {
  static uint8_t model_memory[80];
  tvm_workspace_t workspace = {
      .next_alloc = model_memory,
      .workspace = model_memory,
      .workspace_size = 80,
  };
  tvm_runtime_workspace = &workspace;

  void* block_one = TVMBackendAllocWorkspace(0, 0, 1, 0, 0);
  ASSERT_EQ(block_one, &model_memory[0]);

  void* block_two = TVMBackendAllocWorkspace(0, 0, 2, 0, 0);
  ASSERT_EQ(block_two, &model_memory[16]);

  void* two_blocks = TVMBackendAllocWorkspace(0, 0, 24, 0, 0);
  ASSERT_EQ(two_blocks, &model_memory[32]);

  void* block_three = TVMBackendAllocWorkspace(0, 0, 1, 0, 0);
  ASSERT_EQ(block_three, &model_memory[64]);
}

/*
 * Tests resetting the stack after dealloc
 */
TEST(AOTMemory, Free) {
  static uint8_t model_memory[80];
  tvm_workspace_t workspace = {
      .next_alloc = model_memory,
      .workspace = model_memory,
      .workspace_size = 80,
  };
  tvm_runtime_workspace = &workspace;

  void* block_one = TVMBackendAllocWorkspace(0, 0, 1, 0, 0);
  ASSERT_EQ(block_one, &model_memory[0]);

  void* block_two = TVMBackendAllocWorkspace(0, 0, 1, 0, 0);
  ASSERT_EQ(block_two, &model_memory[16]);
  ASSERT_EQ(0, TVMBackendFreeWorkspace(0, 0, block_two));

  void* two_blocks = TVMBackendAllocWorkspace(0, 0, 2, 0, 0);
  ASSERT_EQ(two_blocks, &model_memory[16]);
  ASSERT_EQ(0, TVMBackendFreeWorkspace(0, 0, two_blocks));

  void* block_three = TVMBackendAllocWorkspace(0, 0, 1, 0, 0);
  ASSERT_EQ(block_three, &model_memory[16]);
}

/*
 * Tests we return NULL if we over allocate
 */
TEST(AOTMemory, OverAllocate) {
  static uint8_t model_memory[72];
  tvm_workspace_t workspace = {
      .next_alloc = model_memory,
      .workspace = model_memory,
      .workspace_size = 72,
  };
  tvm_runtime_workspace = &workspace;

  void* block_one = TVMBackendAllocWorkspace(0, 0, 1, 0, 0);
  ASSERT_EQ(block_one, &model_memory[0]);

  void* block_two = TVMBackendAllocWorkspace(0, 0, 1, 0, 0);
  ASSERT_EQ(block_two, &model_memory[16]);

  void* two_blocks = TVMBackendAllocWorkspace(0, 0, 64, 0, 0);
  ASSERT_EQ(two_blocks, (void*)NULL);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
