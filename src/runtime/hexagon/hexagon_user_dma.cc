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

#include "hexagon_user_dma.h"

#include <algorithm>

#include "hexagon_device_api.h"

namespace tvm {
namespace runtime {
namespace hexagon {

unsigned int HexagonUserDMA::Init() {
  unsigned int status = dmpause() & DM0_STATUS_MASK;
  return status;
}

int HexagonUserDMA::Copy(int queue_id, void* dst, void* src, uint32_t length) {
  // length limited to 24 bits
  if (length > DESC_LENGTH_MASK) {
    return DMA_FAILURE;
  }

  // source address limited to 32 bits
  uint64_t src64 = reinterpret_cast<uint64_t>(src);
  if (!src64 || src64 > DESC_SRC_MASK) {
    return DMA_FAILURE;
  }

  // destination address limited to 32 bits
  uint64_t dst64 = reinterpret_cast<uint64_t>(dst);
  if (!dst64 || dst64 > DESC_DST_MASK) {
    return DMA_FAILURE;
  }

  uint32_t src32 = static_cast<uint32_t>(src64);
  uint32_t dst32 = static_cast<uint32_t>(dst64);

  // get pointer to next descriptor
  dma_desc_2d_t* dma_desc = descriptors_->Next(queue_id);
  if (!dma_desc) {
    return DMA_RETRY;
  }

  // populate descriptor fields
  dma_desc_set_state(dma_desc, DESC_STATE_READY);
  dma_desc_set_next(dma_desc, DMA_NULL_PTR);
  dma_desc_set_length(dma_desc, length);
  dma_desc_set_desctype(dma_desc, DESC_DESCTYPE_1D);
  dma_desc_set_dstcomp(dma_desc, DESC_COMP_NONE);
  dma_desc_set_srccomp(dma_desc, DESC_COMP_NONE);
  dma_desc_set_bypassdst(dma_desc, DESC_BYPASS_OFF);
  dma_desc_set_bypasssrc(dma_desc, DESC_BYPASS_OFF);
  dma_desc_set_order(dma_desc, DESC_ORDER_ORDER);
  dma_desc_set_done(dma_desc, DESC_DONE_INCOMPLETE);
  dma_desc_set_src(dma_desc, src32);
  dma_desc_set_dst(dma_desc, dst32);

  if (first_dma_) {
    // `dmstart` first descriptor
    dmstart(dma_desc);
    first_dma_ = false;
  } else {
    // `dmlink` descriptor to tail descriptor
    dmlink(tail_dma_desc_, dma_desc);
  }

  // update tail
  tail_dma_desc_ = dma_desc;
  return DMA_SUCCESS;
}

void HexagonUserDMA::Wait(int queue_id, uint32_t max_dmas_in_flight) {
  // wait (forever) until max DMAs in flight <= actual DMAs in flight
  while (DMAsInFlight(queue_id) > max_dmas_in_flight) {
  }
}

uint32_t HexagonUserDMA::Poll(int queue_id) { return DMAsInFlight(queue_id); }

uint32_t HexagonUserDMA::DMAsInFlight(int queue_id) {
  dmpoll();  // update DMA engine status
  return descriptors_->InFlight(queue_id);
}

HexagonUserDMA::HexagonUserDMA() {
  // reset DMA engine
  unsigned int status = Init();
  CHECK_EQ(status, DM0_STATUS_IDLE);

  auto desc_in_flight = [](dma_desc_2d_t* dma_desc) {
    unsigned int done = dma_desc_get_done(dma_desc);
    return (done != DESC_DONE_COMPLETE);
  };
  descriptors_ = new QueuedRingBuffer<dma_desc_2d_t>(MAX_DMA_DESCRIPTORS, desc_in_flight);
}

HexagonUserDMA::~HexagonUserDMA() {
  Init();  // stop DMA engine
  delete descriptors_;
}

int hexagon_user_dma_1d_sync(void* dst, void* src, uint32_t length) {
  HexagonUserDMA* user_dma = HexagonDeviceAPI::Global()->UserDMA();

  // One DMA transfer can copy at most DESC_LENGTH_MASK bytes.
  // Make the common case quick.
  if (length <= DESC_LENGTH_MASK) {
    // sync DMA -> `Copy` and then `Wait(0)`
    int ret_val = user_dma->Copy(SYNC_DMA_QUEUE, dst, src, length);
    if (ret_val != DMA_SUCCESS) return ret_val;
    user_dma->Wait(SYNC_DMA_QUEUE, 0);
    return DMA_SUCCESS;
  }

  // Split big transfers into smaller transfers.
  char* cast_src = static_cast<char*>(src);
  char* cast_dst = static_cast<char*>(dst);
  for (uint32_t i = 0; i < length;) {
    // Ensure there is no overflow while updating i
    uint32_t cur_len = std::min<uint32_t>(length - i, DESC_LENGTH_MASK);
    // sync DMA -> `Copy` and then `Wait(0)`
    int ret_val = user_dma->Copy(SYNC_DMA_QUEUE, &cast_dst[i], &cast_src[i], cur_len);
    if (ret_val != DMA_SUCCESS) return ret_val;
    user_dma->Wait(SYNC_DMA_QUEUE, 0);
    // 2 cases for new val for i:
    // 1. length - i <= DESC_LENGTH_MASK (<= MAX_UINT)
    //    new_i = i + (length - i) = length, no more iter
    //            and no overflow (since (length - i) <= (MAX_UINT - i))
    // 2. length - i > DESC_LENGTH_MASK
    //    length > (i + DESC_LENGTH_MASK)
    //    new_i = (i + DESC_LENGTH_MASK)
    //    length > new_i for next iter, we're done
    //    length - i > DESC_LENGTH_MASK
    //    and length <= MAX_UINT,
    //    so MAX_UINT >= length > DESC_LEN_MASK + i
    //    MAX_UINT > (DESC_LEN_MASK + i), so no overflow
    i += cur_len;
  }
  return DMA_SUCCESS;
}
}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
