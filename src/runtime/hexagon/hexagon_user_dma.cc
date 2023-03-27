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

int HexagonUserDMA::Copy(uint32_t queue_id, void* dst, void* src, uint32_t length,
                         bool bypass_cache) {
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

  bool dst_is_ddr = !HexagonDeviceAPI::Global()->VtcmPool()->IsVtcm(dst, length);
  bool src_is_ddr = !HexagonDeviceAPI::Global()->VtcmPool()->IsVtcm(src, length);

  // VTCM -> DDR with bypass enabled
  if (dst_is_ddr && !src_is_ddr && bypass_cache) {
    dma_desc_set_bypassdst(dma_desc, DESC_BYPASS_ON);
  } else {
    dma_desc_set_bypassdst(dma_desc, DESC_BYPASS_OFF);
  }

  // DDR -> VTCM with bypass enabled
  if (src_is_ddr && !dst_is_ddr && bypass_cache) {
    dma_desc_set_bypasssrc(dma_desc, DESC_BYPASS_ON);
  } else {
    dma_desc_set_bypasssrc(dma_desc, DESC_BYPASS_OFF);
  }

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

void HexagonUserDMA::Wait(uint32_t queue_id, uint32_t max_dmas_in_flight) {
  // wait (forever) until max DMAs in flight <= actual DMAs in flight
  while (DMAGroupsInFlight(queue_id) > max_dmas_in_flight) {
  }
}

uint32_t HexagonUserDMA::Poll(uint32_t queue_id) { return DMAGroupsInFlight(queue_id); }

uint32_t HexagonUserDMA::DMAGroupsInFlight(uint32_t queue_id) {
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
  descriptors_ =
      new QueuedRingBuffer<dma_desc_2d_t>(MAX_DMA_QUEUES, MAX_DMA_DESCRIPTORS, desc_in_flight);
}

HexagonUserDMA::~HexagonUserDMA() {
  Init();  // stop DMA engine
  delete descriptors_;
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
