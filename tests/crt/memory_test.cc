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
#include <tvm/runtime/crt/internal/common/memory.h>
#include <tvm/runtime/crt/memory.h>

#include "crt_config.h"
#include "platform.cc"

#define ROUND_UP(qty, modulo) (((qty) + ((modulo)-1)) / (modulo) * (modulo))

static constexpr const unsigned int kTotalPages = 128;
static constexpr const unsigned int kNumUsablePages =
    (sizeof(void*) == 8 ? 95 : (sizeof(void*) == 4 ? 99 : 0));
static constexpr const unsigned int kPageSizeBytesLog = 8;  // 256 byte pages.
static constexpr const unsigned int kMemoryPoolSizeBytes = kTotalPages * (1 << kPageSizeBytesLog);

class MemoryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(raw_memory_pool, 0, sizeof(raw_memory_pool));
    memory_pool = (uint8_t*)(ROUND_UP(((uintptr_t)raw_memory_pool), (1 << kPageSizeBytesLog)));
    MemoryManagerCreate(&mgr, memory_pool, kMemoryPoolSizeBytes, kPageSizeBytesLog);
    ASSERT_EQ(kNumUsablePages, mgr.ptable.max_pages);
  }

  unsigned int AddressToPageNumber(void* a) {
    return (reinterpret_cast<uintptr_t>(a) - reinterpret_cast<uintptr_t>(memory_pool)) >>
           kPageSizeBytesLog;
  }

  uint8_t raw_memory_pool[kMemoryPoolSizeBytes + (1 << kPageSizeBytesLog)];
  uint8_t* memory_pool;
  MemoryManager mgr;
};

#define EXPECT_PAGE(expected, actual) EXPECT_EQ(expected, AddressToPageNumber(actual))

TEST_F(MemoryManagerTest, AllocFreeFifo) {
  EXPECT_EQ(vleak_size, 0);

  for (int i = 0; i < 2; i++) {
    void* ptrs[kNumUsablePages];
    for (size_t idx = 0; idx < kNumUsablePages; idx++) {
      void* a = mgr.Alloc(&mgr, 1);
      if (i == 0) {
        EXPECT_PAGE(idx, a);
      } else {
        EXPECT_PAGE(kNumUsablePages - 1 - idx, a);
      }
      EXPECT_EQ(vleak_size, idx + 1);
      ptrs[idx] = a;
    }

    for (int idx = kNumUsablePages - 1; idx >= 0; idx--) {
      mgr.Free(&mgr, ptrs[idx]);
      EXPECT_EQ(vleak_size, idx);
    }
  }
}

TEST_F(MemoryManagerTest, Realloc) {
  EXPECT_EQ(vleak_size, 0);

  void* a = mgr.Realloc(&mgr, 0, 1);
  EXPECT_PAGE(0, a);
  EXPECT_EQ(vleak_size, 1);

  void* b = mgr.Realloc(&mgr, a, 50);
  EXPECT_PAGE(0, b);
  EXPECT_EQ(vleak_size, 1);

  void* c = mgr.Realloc(&mgr, b, 50 + (1 << kPageSizeBytesLog));
  EXPECT_PAGE(1, c);
  EXPECT_EQ(vleak_size, 1);

  void* d = mgr.Alloc(&mgr, 30);
  EXPECT_PAGE(0, d);
  EXPECT_EQ(vleak_size, 2);

  void* e = mgr.Realloc(&mgr, c, (50 + (2 << kPageSizeBytesLog)));
  EXPECT_PAGE(3, e);
  EXPECT_EQ(vleak_size, 2);

  void* f = mgr.Alloc(&mgr, 30);
  EXPECT_PAGE(1, f);
  EXPECT_EQ(vleak_size, 3);

  mgr.Free(&mgr, f);
  EXPECT_EQ(vleak_size, 2);

  mgr.Free(&mgr, e);
  EXPECT_EQ(vleak_size, 1);

  mgr.Free(&mgr, e);
  EXPECT_EQ(vleak_size, 0);

  void* g = mgr.Alloc(&mgr, 1);
  EXPECT_PAGE(1, g);
  EXPECT_EQ(vleak_size, 1);

  mgr.Free(&mgr, g);
  EXPECT_EQ(vleak_size, 0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
