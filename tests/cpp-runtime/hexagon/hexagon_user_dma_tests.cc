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

//#include <chrono>
#include <gtest/gtest.h>
#include <time.h>

#include "../src/runtime/hexagon/hexagon_user_dma.h"

using namespace tvm::runtime;
using namespace tvm::runtime::hexagon;

class HexagonUserDMATest : public ::testing::Test {
  void SetUp() override {
    src = malloc(length);
    dst = malloc(length);
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    src_char = static_cast<char*>(src);
    dst_char = static_cast<char*>(dst);
    for (uint32_t i = 0; i < length; ++i) {
      src_char[i] = 1;
      dst_char[i] = 0;
    }
  }
  void TearDown() override {
    free(src);
    free(dst);
  }

 public:
  int ret{-1};
  void* src{nullptr};
  void* dst{nullptr};
  char* src_char{nullptr};
  char* dst_char{nullptr};
  uint32_t length{0x400000};  // 4MB
};

TEST_F(HexagonUserDMATest, wait) {
  HexagonUserDMA::Get().Wait(0);
  HexagonUserDMA::Get().Wait(10);
}

TEST_F(HexagonUserDMATest, poll) { ASSERT_EQ(HexagonUserDMA::Get().Poll(), 0); }

TEST_F(HexagonUserDMATest, bad_copy) {
  uint64_t bigaddr = 0x100000000;
  void* src64 = reinterpret_cast<void*>(bigaddr);
  void* dst64 = reinterpret_cast<void*>(bigaddr);
  uint32_t biglength = 0x1000000;
  ASSERT_NE(HexagonUserDMA::Get().Copy(dst64, src, length), DMA_SUCCESS);
  ASSERT_NE(HexagonUserDMA::Get().Copy(dst, src64, length), DMA_SUCCESS);
  ASSERT_NE(HexagonUserDMA::Get().Copy(dst, src, biglength), DMA_SUCCESS);
}

TEST_F(HexagonUserDMATest, sync_dma) {
  // kick off 1 DMA
  ret = HexagonUserDMA::Get().Copy(dst, src, length);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // wait for DMA to complete
  HexagonUserDMA::Get().Wait(0);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }
}

TEST_F(HexagonUserDMATest, async_dma) {
  // kick off 10x duplicate DMAs
  for (uint32_t i = 0; i < 10; ++i) {
    ret = HexagonUserDMA::Get().Copy(dst, src, length);
    ASSERT_EQ(ret, DMA_SUCCESS);
  }

  // verify at least 1 DMA in flight
  ASSERT_GT(HexagonUserDMA::Get().Poll(), 0);

  // wait for at least 1 DMA to complete
  HexagonUserDMA::Get().Wait(9);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }
}

TEST_F(HexagonUserDMATest, pipeline) {
  // auto start = std::chrono::high_resolution_clock::now();
  time_t start = time(nullptr);

  // sync DMA
  ret = HexagonUserDMA::Get().Copy(dst, src, length);
  HexagonUserDMA::Get().Wait(0);

  // compute
  for (uint32_t i = 0; i < length; ++i) {
    dst_char[i]++;
  }

  // auto end = std::chrono::high_resolution_clock::now();
  // auto sync_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  time_t end = time(nullptr);
  double sync_ns = difftime(start, end);

  // verify
  ASSERT_EQ(ret, 0);
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(dst_char[i], 2);
  }

  // start = std::chrono::high_resolution_clock::now();
  start = time(nullptr);

  uint32_t pipeline_depth = 4;
  uint32_t pipeline_length = length / pipeline_depth;

  for (uint32_t i = 0; i < pipeline_depth; ++i) {
    ret |= HexagonUserDMA::Get().Copy(dst_char + i * pipeline_length,
                                      src_char + i * pipeline_length, pipeline_length);
  }

  HexagonUserDMA::Get().Wait(3);
  for (uint32_t i = 0; i < pipeline_length; ++i) {
    dst_char[i]++;
  }

  HexagonUserDMA::Get().Wait(2);
  for (uint32_t i = pipeline_length; i < 2 * pipeline_length; ++i) {
    dst_char[i]++;
  }

  HexagonUserDMA::Get().Wait(1);
  for (uint32_t i = 2 * pipeline_length; i < 3 * pipeline_length; ++i) {
    dst_char[i]++;
  }

  HexagonUserDMA::Get().Wait(0);
  for (uint32_t i = 3 * pipeline_length; i < 4 * pipeline_length; ++i) {
    dst_char[i]++;
  }

  // end = std::chrono::high_resolution_clock::now();
  // auto async_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  end = time(nullptr);
  double async_ns = difftime(start, end);

  // verify
  ASSERT_EQ(ret, 0);
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(dst_char[i], 2);
  }

  // TODO: time/chrono not working, returns 0
  ASSERT_LE(async_ns, sync_ns);
}