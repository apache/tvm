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
  void SetUp() override {
    vtcm_pool = HexagonDeviceAPI::Global()->VtcmPool();
    max_bytes = vtcm_pool->VtcmAllocatedBytes();
    device_bytes = vtcm_pool->VtcmDeviceBytes();
  }
  void TearDown() override {}

 public:
  HexagonVtcmPool* vtcm_pool;
  size_t max_bytes;
  size_t device_bytes;
  size_t four_k_block = 4096;
  size_t two_k_block = 2048;
  size_t one_k_block = 1024;
  size_t min_bytes = 128;
};

TEST_F(HexagonVtcmPoolTest, basic) {
  void* ptr;
  void* ptr2;

  CHECK(device_bytes >= max_bytes) << "VTCM device size " << device_bytes
                                   << " not greater than or equal to allocated size " << max_bytes;

  ptr = vtcm_pool->Allocate(max_bytes);
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7FF) == 0)
      << "Must be multiple of 2k " << ptr << " " << max_bytes;
  vtcm_pool->Free(ptr, max_bytes);

  ptr = vtcm_pool->Allocate(two_k_block);
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7FF) == 0)
      << "Must be multiple of 2k " << ptr << " " << two_k_block;
  vtcm_pool->Free(ptr, two_k_block);

  ptr = vtcm_pool->Allocate(one_k_block);
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7F) == 0)
      << "Must be multiple of 128 " << ptr << " " << one_k_block;
  vtcm_pool->Free(ptr, one_k_block);

  ptr = vtcm_pool->Allocate(min_bytes);
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7F) == 0)
      << "Must be multiple of 128 " << ptr << " " << min_bytes;

  ptr2 = vtcm_pool->Allocate(one_k_block);
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7F) == 0)
      << "Must be multiple of 128 " << ptr2 << " " << one_k_block;
  vtcm_pool->Free(ptr, min_bytes);
  vtcm_pool->Free(ptr2, one_k_block);

  EXPECT_THROW(ptr = vtcm_pool->Allocate(1), InternalError);
}

TEST_F(HexagonVtcmPoolTest, small_allocations) {
  void* ptr1;
  void* ptr2;
  void* ptr3;
  void* ptr4;

  // Allocate small chunk from the back
  ptr1 = vtcm_pool->Allocate(min_bytes);

  // Allocate from the front
  ptr2 = vtcm_pool->Allocate(two_k_block);

  // Allocate the rest
  ptr3 = vtcm_pool->Allocate(max_bytes - min_bytes - two_k_block);

  // Should be no more memory left
  EXPECT_THROW(ptr4 = vtcm_pool->Allocate(min_bytes), InternalError);

  vtcm_pool->Free(ptr1, min_bytes);
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, max_bytes - min_bytes - two_k_block);

  // Make sure at the end we have the full amount available again
  ptr4 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr4, max_bytes);
}

TEST_F(HexagonVtcmPoolTest, no_free_vtcm) {
  void* ptr = vtcm_pool->Allocate(max_bytes);
  EXPECT_THROW(vtcm_pool->Allocate(min_bytes), InternalError);
  vtcm_pool->Free(ptr, max_bytes);
}

TEST_F(HexagonVtcmPoolTest, not_enough_free_vtcm) {
  void* ptr = vtcm_pool->Allocate(max_bytes - two_k_block);
  EXPECT_THROW(vtcm_pool->Allocate(two_k_block * 2), InternalError);
  vtcm_pool->Free(ptr, max_bytes - two_k_block);
}

TEST_F(HexagonVtcmPoolTest, free_with_wrong_size) {
  void* ptr = vtcm_pool->Allocate(two_k_block * 2);
  EXPECT_THROW(vtcm_pool->Free(ptr, two_k_block), InternalError);
  vtcm_pool->Free(ptr, two_k_block * 2);
}

TEST_F(HexagonVtcmPoolTest, free_alloc_combinations) {
  void* ptr1;
  void* ptr2;
  void* ptr3;
  void* ptr4;
  void* new_ptr;
  size_t max_less_3_blocks = max_bytes - (3 * two_k_block);
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

  // Make sure at the end we have the full amount available again
  ptr4 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr4, max_bytes);
}

TEST_F(HexagonVtcmPoolTest, find_allocation) {
  void* ptr1;
  void* ptr2;
  void* ptr3;

  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);

  // Free the first allocation
  vtcm_pool->Free(ptr1, two_k_block);

  // Allocate a new larger block to initiate search and ensure
  // it succeeds despite there not being a match in the first free block.
  ptr3 = vtcm_pool->Allocate(four_k_block);

  // Clean up the ptrs
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, four_k_block);

  // Make sure at the end we have the full amount available again
  ptr1 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr1, max_bytes);
}

TEST_F(HexagonVtcmPoolTest, find_smallest_allocation_combinations) {
  void* ptr1;
  void* ptr2;
  void* ptr3;
  void* ptr4;
  void* new_ptr;

  ptr1 = vtcm_pool->Allocate(two_k_block);
  ptr2 = vtcm_pool->Allocate(two_k_block);
  ptr3 = vtcm_pool->Allocate(four_k_block);
  ptr4 = vtcm_pool->Allocate(four_k_block);

  // Fragment memory allocations.
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, four_k_block);

  // Reallocate memory allocations and ensure that the smallest free allocations are used.
  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr2);

  new_ptr = vtcm_pool->Allocate(two_k_block);
  CHECK(new_ptr == ptr3);

  vtcm_pool->Free(ptr1, two_k_block);
  vtcm_pool->Free(ptr2, two_k_block);
  vtcm_pool->Free(ptr3, two_k_block);
  vtcm_pool->Free(ptr4, four_k_block);

  // Rerun the same test for non 2k aligned allocations.
  ptr1 = vtcm_pool->Allocate(min_bytes);
  ptr2 = vtcm_pool->Allocate(min_bytes);
  ptr3 = vtcm_pool->Allocate(one_k_block);
  ptr4 = vtcm_pool->Allocate(one_k_block);

  // Fragment memory allocations.
  vtcm_pool->Free(ptr2, min_bytes);
  vtcm_pool->Free(ptr3, one_k_block);

  // Reallocate memory allocations and ensure that the smallest free allocations are used.
  new_ptr = vtcm_pool->Allocate(min_bytes);
  CHECK(new_ptr == ptr2);

  new_ptr = vtcm_pool->Allocate(one_k_block);
  CHECK(new_ptr == ptr3);

  vtcm_pool->Free(ptr1, min_bytes);
  vtcm_pool->Free(ptr2, min_bytes);
  vtcm_pool->Free(ptr3, one_k_block);
  vtcm_pool->Free(ptr4, one_k_block);

  // Make sure at the end we have the full amount available again
  ptr4 = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr4, max_bytes);
}

// Test alignment edge cases allocating through HexagonBuffer
TEST_F(HexagonVtcmPoolTest, vtcm_alignment) {
  std::unique_ptr<HexagonBufferManager> test_hexbuffs = std::make_unique<HexagonBufferManager>();
  void* ptr;

  // Invalid alignments
  EXPECT_THROW(test_hexbuffs->AllocateHexagonBuffer(min_bytes, 128 + 1, String("global")),
               InternalError);
  EXPECT_THROW(test_hexbuffs->AllocateHexagonBuffer(min_bytes, 2048 + 1, String("global")),
               InternalError);

  // Valid alignments, sizes need to be adjusted
  ptr = test_hexbuffs->AllocateHexagonBuffer(1, 128, String("global"));
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7F) == 0) << "Must be multiple of 128 " << ptr;

  ptr = test_hexbuffs->AllocateHexagonBuffer(127, 128, String("global"));
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7F) == 0) << "Must be multiple of 128 " << ptr;

  ptr = test_hexbuffs->AllocateHexagonBuffer(129, 128, String("global"));
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7F) == 0) << "Must be multiple of 128 " << ptr;

  ptr = test_hexbuffs->AllocateHexagonBuffer(1, 2048, String("global"));
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7FF) == 0) << "Must be multiple of 2k " << ptr;

  ptr = test_hexbuffs->AllocateHexagonBuffer(2047, 2048, String("global"));
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7FF) == 0) << "Must be multiple of 2k " << ptr;

  ptr = test_hexbuffs->AllocateHexagonBuffer(2049, 2048, String("global"));
  CHECK((reinterpret_cast<uintptr_t>(ptr) & 0x7FF) == 0) << "Must be multiple of 2k " << ptr;

  test_hexbuffs.reset();

  // Make sure at the end we have the full amount available again
  ptr = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr, max_bytes);
}
