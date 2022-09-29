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
    vtcm_pool = std::make_unique<HexagonVtcmPool>();
  }
  void TearDown() override {
    vtcm_pool.reset();
  }

 public:
  std::unique_ptr<HexagonVtcmPool> vtcm_pool;
};

TEST_F(HexagonVtcmPoolTest, basic) {
  void* ptr;
  size_t max_bytes = vtcm_pool->TotalBytes();
  size_t two_k_block = 2048;
  size_t one_k_block = 1024;
  ptr = vtcm_pool->Allocate(max_bytes);
  vtcm_pool->Free(ptr, max_bytes);
  ptr = vtcm_pool->Allocate(two_k_block);
  vtcm_pool->Free(ptr, two_k_block);
  ptr = vtcm_pool->Allocate(one_k_block);
  vtcm_pool->Free(ptr, one_k_block);
}

// TEST_F(HexagonVtcmPoolTest, basic) {
//   void* ptr;
//   size_t max_bytes = 1024*1024;
//   size_t two_k_block = 2048;
//   size_t one_k_block = 1024;
//   ptr = vtcm_pool->Allocate(max_bytes);
//   vtcm_pool->Free(ptr, max_bytes);
//   ptr = vtcm_pool->Allocate(two_k_block);
//   vtcm_pool->Free(ptr, two_k_block);
//   ptr = vtcm_pool->Allocate(one_k_block);
//   vtcm_pool->Free(ptr, one_k_block);
// }
