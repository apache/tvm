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

#include "../src/runtime/hexagon/hexagon_device_api.h"

using namespace tvm::runtime;
using namespace tvm::runtime::hexagon;

class HexagonVtcmPoolTest : public ::testing::Test {
  void SetUp() override { vtcm_pool = HexagonDeviceAPI::Global()->VtcmPool(); }
  void TearDown() override {}

 public:
  HexagonVtcmPool* vtcm_pool;
};

TEST_F(HexagonVtcmPoolTest, basic) {
  void* ptr;
  size_t max_bytes = vtcm_pool->TotalBytes();
  size_t two_k_block = 2048;
  size_t one_k_block = 1024;
  size_t one_byte_block = 1;
  ptr = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr, max_bytes);
  ptr = vtcm_pool->Allocate(two_k_block);
  vtcm_pool->Free(ptr, two_k_block);
  ptr = vtcm_pool->Allocate(one_k_block);
  vtcm_pool->Free(ptr, one_k_block);
  ptr = vtcm_pool->Allocate(one_byte_block);
  vtcm_pool->Free(ptr, one_byte_block);
}

TEST_F(HexagonVtcmPoolTest, no_free_vtcm) {
  void* ptr;
  size_t max_bytes = vtcm_pool->TotalBytes();
  ptr = vtcm_pool->Allocate(max_bytes);
  EXPECT_THROW(vtcm_pool->Allocate(1), InternalError);
  vtcm_pool->Free(ptr, max_bytes);
}

TEST_F(HexagonVtcmPoolTest, not_enough_free_vtcm) {
  void* ptr;
  size_t max_bytes = vtcm_pool->TotalBytes();
  size_t two_k_block = 2048;
  ptr = vtcm_pool->Allocate(max_bytes - two_k_block);
  EXPECT_THROW(vtcm_pool->Allocate(two_k_block * 2), InternalError);
  vtcm_pool->Free(ptr, max_bytes - two_k_block);
}

TEST_F(HexagonVtcmPoolTest, free_with_wrong_size) {
  void* ptr;
  size_t two_k_block = 2048;
  ptr = vtcm_pool->Allocate(two_k_block * 2);
  EXPECT_THROW(vtcm_pool->Free(ptr, two_k_block), InternalError);
  vtcm_pool->Free(ptr, two_k_block * 2);
}

TEST_F(HexagonVtcmPoolTest, free_alloc_combinations) {
  void* ptr1;
  void* ptr2;
  void* ptr3;
  void* ptr4;
  void* new_ptr;
  size_t two_k_block = 2048;
  size_t max_less_3_blocks = vtcm_pool->TotalBytes() - (3 * two_k_block);
  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);
  ptr3 = vtcm_pool->Allocate(two_k_block);
  ptr4 = vtcm_pool->Allocate(max_less_3_blocks);

  // Make sure pointers are 2k apart from each other
  CHECK(static_cast<char*>(ptr1) + two_k_block == static_cast<char*>(ptr2));
  CHECK(static_cast<char*>(ptr2) + two_k_block == static_cast<char*>(ptr3));
  CHECK(static_cast<char*>(ptr3) + two_k_block == static_cast<char*>(ptr4));

  // Free 2, realloc it, make sure it is the same as before
  vtcm_pool->Free(ptr2, two_k_block);
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr2);

  // Free 1 and 2, re-alloc and make sure they are the same
  vtcm_pool->Free(ptr1, two_k_block);
  vtcm_pool->Free(ptr2, two_k_block);
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr1);
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr2);

  // Exercise different deletion scenarios
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, two_k_block);
  vtcm_pool->Free(ptr4, max_less_3_blocks);
  vtcm_pool->Free(ptr1, two_k_block);

  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);
  ptr3 = vtcm_pool->Allocate(two_k_block);
  vtcm_pool->Free(ptr1, two_k_block);
  vtcm_pool->Free(ptr3, two_k_block);
  vtcm_pool->Free(ptr2, two_k_block);

  // Make sure at the end we have the full amount
  // available again
  ptr4 = vtcm_pool->Allocate(max_less_3_blocks);
  vtcm_pool->Free(ptr4, max_less_3_blocks);
}
