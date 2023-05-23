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

class HexagonUserDMATest : public ::testing::Test {
  void SetUp() override {
    user_dma = HexagonDeviceAPI::Global()->UserDMA();
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
  HexagonUserDMA* user_dma;
  int ret = 0;
  uint32_t queue_id = 0;
  void* src = nullptr;
  void* dst = nullptr;
  char* src_char = nullptr;
  char* dst_char = nullptr;
  uint32_t length = 0x4000;  // 16KB
  const bool ENABLE_BYPASS = true;
  const bool DISABLE_BYPASS = false;
  Optional<String> global_scope{"global"};
  Optional<String> global_vtcm_scope{"global.vtcm"};
};

TEST_F(HexagonUserDMATest, wait) {
  user_dma->Wait(queue_id, 0);
  user_dma->Wait(queue_id, 10);
}

TEST_F(HexagonUserDMATest, poll) { ASSERT_EQ(user_dma->Poll(queue_id), 0); }

TEST_F(HexagonUserDMATest, bad_copy) {
  uint64_t bigaddr = 0x100000000;
  void* src64 = reinterpret_cast<void*>(bigaddr);
  void* dst64 = reinterpret_cast<void*>(bigaddr);
  uint32_t biglength = 0x1000000;
  ASSERT_NE(user_dma->Copy(queue_id, dst64, src, length, DISABLE_BYPASS), DMA_SUCCESS);
  ASSERT_NE(user_dma->Copy(queue_id, dst, src64, length, DISABLE_BYPASS), DMA_SUCCESS);
  ASSERT_NE(user_dma->Copy(queue_id, dst, src, biglength, DISABLE_BYPASS), DMA_SUCCESS);
}

TEST_F(HexagonUserDMATest, sync_dma) {
  // kick off 1 DMA
  ret = user_dma->Copy(queue_id, dst, src, length, DISABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // wait for DMA to complete
  user_dma->Wait(queue_id, 0);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }
}

TEST_F(HexagonUserDMATest, async_dma_wait) {
  // kick off 10x duplicate DMAs
  for (uint32_t i = 0; i < 10; ++i) {
    ret = user_dma->Copy(queue_id, dst, src, length, DISABLE_BYPASS);
    ASSERT_EQ(ret, DMA_SUCCESS);
  }

  // wait for at least 1 DMA to complete
  user_dma->Wait(queue_id, 9);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }

  // empty the DMA queue
  user_dma->Wait(queue_id, 0);
}

TEST_F(HexagonUserDMATest, async_dma_poll) {
  // kick off 10x duplicate DMAs
  for (uint32_t i = 0; i < 10; ++i) {
    ret = user_dma->Copy(queue_id, dst, src, length, DISABLE_BYPASS);
    ASSERT_EQ(ret, DMA_SUCCESS);
  }

  // poll until at least 1 DMA is complete
  while (user_dma->Poll(queue_id) == 10) {
  };

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }

  // empty the DMA queue
  user_dma->Wait(queue_id, 0);
}

TEST_F(HexagonUserDMATest, pipeline) {
  uint32_t pipeline_depth = 4;
  uint32_t pipeline_length = length / pipeline_depth;

  for (uint32_t i = 0; i < pipeline_depth; ++i) {
    ret |= user_dma->Copy(queue_id, dst_char + i * pipeline_length, src_char + i * pipeline_length,
                          pipeline_length, DISABLE_BYPASS);
  }

  user_dma->Wait(queue_id, 3);
  for (uint32_t i = 0; i < pipeline_length; ++i) {
    dst_char[i]++;
  }

  user_dma->Wait(queue_id, 2);
  for (uint32_t i = pipeline_length; i < 2 * pipeline_length; ++i) {
    dst_char[i]++;
  }

  user_dma->Wait(queue_id, 1);
  for (uint32_t i = 2 * pipeline_length; i < 3 * pipeline_length; ++i) {
    dst_char[i]++;
  }

  user_dma->Wait(queue_id, 0);
  for (uint32_t i = 3 * pipeline_length; i < 4 * pipeline_length; ++i) {
    dst_char[i]++;
  }

  // verify
  ASSERT_EQ(ret, DMA_SUCCESS);
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(2, dst_char[i]);
  }
}

TEST_F(HexagonUserDMATest, pipeline_write_queue) {
  int write_queue = queue_id + 1;
  uint32_t pipeline_depth = 4;
  uint32_t pipeline_length = length / pipeline_depth;

  for (uint32_t i = 0; i < pipeline_depth; ++i) {
    ret |= user_dma->Copy(queue_id, dst_char + i * pipeline_length, src_char + i * pipeline_length,
                          pipeline_length, DISABLE_BYPASS);
  }

  user_dma->Wait(queue_id, 3);
  for (uint32_t i = 0; i < pipeline_length; ++i) {
    dst_char[i]++;
  }
  ret |= user_dma->Copy(write_queue, src_char, dst_char, pipeline_length, DISABLE_BYPASS);

  user_dma->Wait(queue_id, 2);
  for (uint32_t i = pipeline_length; i < 2 * pipeline_length; ++i) {
    dst_char[i]++;
  }
  ret |= user_dma->Copy(write_queue, src_char + pipeline_length, dst_char + pipeline_length,
                        pipeline_length, DISABLE_BYPASS);

  user_dma->Wait(queue_id, 1);
  for (uint32_t i = 2 * pipeline_length; i < 3 * pipeline_length; ++i) {
    dst_char[i]++;
  }
  ret |= user_dma->Copy(write_queue, src_char + 2 * pipeline_length, dst_char + 2 * pipeline_length,
                        pipeline_length, DISABLE_BYPASS);

  user_dma->Wait(queue_id, 0);
  for (uint32_t i = 3 * pipeline_length; i < 4 * pipeline_length; ++i) {
    dst_char[i]++;
  }
  ret |= user_dma->Copy(write_queue, src_char + 3 * pipeline_length, dst_char + 3 * pipeline_length,
                        pipeline_length, DISABLE_BYPASS);
  user_dma->Wait(write_queue, 0);

  // verify
  ASSERT_EQ(ret, DMA_SUCCESS);
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(2, dst_char[i]);
    ASSERT_EQ(2, src_char[i]);
  }
}

TEST_F(HexagonUserDMATest, overflow_ring_buffer) {
  uint32_t number_of_dmas = 0x400;  // 1k
  uint32_t length_of_each_dma = length / number_of_dmas;

  for (uint32_t i = 0; i < number_of_dmas; ++i) {
    do {
      ret = user_dma->Copy(queue_id, dst_char + i * length_of_each_dma,
                           src_char + i * length_of_each_dma, length_of_each_dma, DISABLE_BYPASS);
    } while (ret == DMA_RETRY);
    ASSERT_EQ(ret, DMA_SUCCESS);
  }

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }
}

TEST_F(HexagonUserDMATest, sync_dma_bypass) {
  HexagonBuffer srchb(length, kHexagonAllocAlignment, global_scope);
  HexagonBuffer dsthb(length, kHexagonAllocAlignment, global_scope);
  HexagonBuffer vtcmhb(length, kHexagonAllocAlignment, global_vtcm_scope);

  // init src, dst HexagonBuffers
  srchb.CopyFrom(src, length);
  dsthb.CopyFrom(dst, length);

  // DDR src -> VTCM
  ret = user_dma->Copy(queue_id, vtcmhb.GetPointer(), srchb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // VTCM -> DDR dst
  ret = user_dma->Copy(queue_id, dsthb.GetPointer(), vtcmhb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // wait for DMAs to complete
  user_dma->Wait(queue_id, 0);

  // copy answer from dst HexagonBuffer
  dsthb.CopyTo(dst, length);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }
}

TEST_F(HexagonUserDMATest, sync_dma_bypass_vtcm_to_vtcm) {
  HexagonBuffer srchb(length, kHexagonAllocAlignment, global_scope);
  HexagonBuffer dsthb(length, kHexagonAllocAlignment, global_scope);
  HexagonBuffer vtcm1hb(length, kHexagonAllocAlignment, global_vtcm_scope);
  HexagonBuffer vtcm2hb(length, kHexagonAllocAlignment, global_vtcm_scope);

  // init src, dst HexagonBuffers
  srchb.CopyFrom(src, length);
  dsthb.CopyFrom(dst, length);

  // DDR src -> VTCM
  ret = user_dma->Copy(queue_id, vtcm1hb.GetPointer(), srchb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // VTCM -> VTCM
  // NOTE: Cache bypass is disabled for VTCM -> VTCM transfers
  ret =
      user_dma->Copy(queue_id, vtcm2hb.GetPointer(), vtcm1hb.GetPointer(), length, DISABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // VTCM -> DDR dst
  ret = user_dma->Copy(queue_id, dsthb.GetPointer(), vtcm2hb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // wait for DMAs to complete
  user_dma->Wait(queue_id, 0);

  // copy answer from dst HexagonBuffer
  dsthb.CopyTo(dst, length);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }
}

TEST_F(HexagonUserDMATest, sync_dma_bypass_) {
  HexagonBuffer srchb(length, kHexagonAllocAlignment, global_scope);
  HexagonBuffer dsthb(length, kHexagonAllocAlignment, global_scope);
  HexagonBuffer vtcmhb(length, kHexagonAllocAlignment, global_vtcm_scope);

  // init src, dst HexagonBuffers
  srchb.CopyFrom(src, length);
  dsthb.CopyFrom(dst, length);

  // DDR src -> VTCM
  ret = user_dma->Copy(queue_id, vtcmhb.GetPointer(), srchb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // VTCM -> DDR dst
  ret = user_dma->Copy(queue_id, dsthb.GetPointer(), vtcmhb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // wait for DMAs to complete
  user_dma->Wait(queue_id, 0);

  // copy answer from dst HexagonBuffer
  dsthb.CopyTo(dst, length);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }

  // change src data
  for (uint32_t i = 0; i < length; ++i) {
    src_char[i] = 2;
  }

  // copy new src data to HexagonBuffer
  srchb.CopyFrom(src, length);

  // DDR src -> VTCM
  ret = user_dma->Copy(queue_id, vtcmhb.GetPointer(), srchb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // VTCM -> DDR dst
  ret = user_dma->Copy(queue_id, dsthb.GetPointer(), vtcmhb.GetPointer(), length, ENABLE_BYPASS);
  ASSERT_EQ(ret, DMA_SUCCESS);

  // wait for DMAs to complete
  user_dma->Wait(queue_id, 0);

  // copy answer from dst HexagonBuffer
  dsthb.CopyTo(dst, length);

  // verify
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(src_char[i], dst_char[i]);
  }
}
