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
#include <tvm/runtime/crt/internal/memory/page_allocator.h>
#include <tvm/runtime/crt/page_allocator.h>

#include "crt_config.h"

#define ROUND_UP(qty, modulo) (((qty) + ((modulo)-1)) / (modulo) * (modulo))

static constexpr const unsigned int kTotalPages = 128;
static constexpr const unsigned int kNumUsablePages =
    (sizeof(void*) == 8 ? 94 : (sizeof(void*) == 4 ? 99 : 0));
static constexpr const unsigned int kPageSizeBytesLog = 8;  // 256 byte pages.
static constexpr const unsigned int kMemoryPoolSizeBytes = kTotalPages * (1 << kPageSizeBytesLog);

class PageAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memset(raw_memory_pool, 0, sizeof(raw_memory_pool));
    memory_pool = reinterpret_cast<uint8_t*>(
        ROUND_UP(((uintptr_t)raw_memory_pool), (1 << kPageSizeBytesLog)));
    PageMemoryManagerCreate(&interface, memory_pool, kMemoryPoolSizeBytes, kPageSizeBytesLog);
    mgr = reinterpret_cast<MemoryManager*>(interface);
    ASSERT_EQ(kNumUsablePages, mgr->ptable.max_pages);
    dev_ = {kDLCPU, 0};
  }

  unsigned int AddressToPageNumber(void* a) {
    return (reinterpret_cast<uintptr_t>(a) - reinterpret_cast<uintptr_t>(memory_pool)) >>
           kPageSizeBytesLog;
  }

  uint8_t raw_memory_pool[kMemoryPoolSizeBytes + (1 << kPageSizeBytesLog)];
  uint8_t* memory_pool;
  MemoryManagerInterface* interface;
  MemoryManager* mgr;
  DLDevice dev_;
};

#define EXPECT_PAGE(expected, actual) EXPECT_EQ(expected, AddressToPageNumber(actual))

TEST_F(PageAllocatorTest, AllocFreeFifo) {
  EXPECT_EQ(interface->vleak_size, 0);

  for (int i = 0; i < 2; i++) {
    void* ptrs[kNumUsablePages];
    for (size_t idx = 0; idx < kNumUsablePages; idx++) {
      void* a;
      EXPECT_EQ(interface->Allocate(interface, 1, dev_, &a), kTvmErrorNoError);
      if (i == 0) {
        EXPECT_PAGE(idx, a);
      } else {
        EXPECT_PAGE(kNumUsablePages - 1 - idx, a);
      }
      EXPECT_EQ(static_cast<size_t>(interface->vleak_size), idx + 1);
      ptrs[idx] = a;
    }

    for (int idx = kNumUsablePages - 1; idx >= 0; idx--) {
      interface->Free(interface, ptrs[idx], dev_);
      EXPECT_EQ(interface->vleak_size, idx);
    }
  }
}
