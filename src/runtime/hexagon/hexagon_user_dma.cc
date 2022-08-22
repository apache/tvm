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

#include "hexagon_common.h"
#include "hexagon_user_dma_descriptors.h"
#include "hexagon_user_dma_instructions.h"
#include "hexagon_user_dma_registers.h"

namespace tvm {
namespace runtime {
namespace hexagon {

unsigned int HexagonUserDMA::Init() {
  unsigned int status = dmpause() & DM0_STATUS_MASK;
  return status;
}

int HexagonUserDMA::Copy(void* dst, void* src, uint32_t length) {
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

  // check if the next DMA descriptor will overwrite an in flight DMA descriptor
  // if this is the first DMA there is nothting to check
  if (!first_dma_) {
    // update the ID of the oldest DMA descriptor in flight
    DMAsInFlight();
    // calcultate whether there are DMA descriptors in flight
    bool dma_desc_in_flight = id_next_dma_desc_ != id_oldest_dma_desc_in_flight_;
    // calculate whether the next DMA descriptor will overwrite the oldest DMA descriptor in flight
    bool same_ring_buff_index = (id_next_dma_desc_ % dma_desc_ring_buff_size_) ==
                                (id_oldest_dma_desc_in_flight_ % dma_desc_ring_buff_size_);
    // fail if there are DMA descriptors in flight
    // and the next DMA descriptor overwrites the oldest DMA descriptor in flight
    if (dma_desc_in_flight && same_ring_buff_index) {
      return DMA_FAILURE;
    }
  }

  // get pointer to next DMA descriptor
  void* dma_desc = GetDescriptorAddr(id_next_dma_desc_);

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
    void* tail = GetDescriptorAddr(id_next_dma_desc_ - 1);
    dmlink(tail, dma_desc);
  }

  // update the ID of the next DMA descriptor
  id_next_dma_desc_++;

  return DMA_SUCCESS;
}

void HexagonUserDMA::Wait(uint32_t max_dmas_in_flight) {
  // wait (forever) until max DMAs in flight <= actual DMAs in flight
  while (DMAsInFlight() > max_dmas_in_flight) {
  }
}

uint32_t HexagonUserDMA::Poll() { return DMAsInFlight(); }

uint32_t HexagonUserDMA::DMAsInFlight() {
  // poll DMA engine to make sure DMA status is current
  dmpoll();

  // find the oldest DMA descriptor in flight
  // total number of DMA descriptors in flight == ID of the next DMA descriptor
  for (; id_oldest_dma_desc_in_flight_ < id_next_dma_desc_; ++id_oldest_dma_desc_in_flight_) {
    // read the `done` bit from the DMA descriptor and stop if incomplete
    unsigned int done = dma_desc_get_done(GetDescriptorAddr(id_oldest_dma_desc_in_flight_));
    if (done == DESC_DONE_INCOMPLETE) {
      break;
    }
  }

  // total DMA descriptors in flight = total number DMA desc - ID of the oldest DMA desc in flight
  // note that these two IDs are equivalent when no DMA descriptors are in flight
  return id_next_dma_desc_ - id_oldest_dma_desc_in_flight_;
}

void* HexagonUserDMA::GetDescriptorAddr(uint32_t dma_desc_id) {
  return static_cast<char*>(dma_desc_ring_buff_) +
         DMA_DESC_2D_SIZE * (dma_desc_id % dma_desc_ring_buff_size_);
}

HexagonUserDMA::HexagonUserDMA() {
  // reset DMA engine
  unsigned int status = Init();
  CHECK_EQ(status, DM0_STATUS_IDLE);

  // allocate memory for ring buffer storage for all DMA descriptors
  int ret = posix_memalign(&dma_desc_ring_buff_, DMA_DESC_2D_SIZE,
                           DMA_DESC_2D_SIZE * dma_desc_ring_buff_size_);
  CHECK_EQ(ret, 0);
  CHECK_NE(dma_desc_ring_buff_, nullptr);
}

HexagonUserDMA::~HexagonUserDMA() {
  // stop the DMA engine
  Init();
  free(dma_desc_ring_buff_);
}

int hexagon_user_dma_1d_sync(void* dst, void* src, uint32_t length) {
  // One DMA transfer can copy at most DESC_LENGTH_MASK bytes.
  // Make the common case quick.
  if (length <= DESC_LENGTH_MASK) {
    // sync DMA -> `Copy` and then `Wait(0)`
    int ret_val = HexagonUserDMA::Get().Copy(dst, src, length);
    if (ret_val != DMA_SUCCESS) return ret_val;
    HexagonUserDMA::Get().Wait(0);
    return DMA_SUCCESS;
  }

  // Split big transfers into smaller transfers.
  char* cast_src = static_cast<char*>(src);
  char* cast_dst = static_cast<char*>(dst);
  for (uint32_t i = 0; i < length;) {
    // Ensure there is no overflow while updating i
    uint32_t cur_len = std::min<uint32_t>(length - i, DESC_LENGTH_MASK);
    // sync DMA -> `Copy` and then `Wait(0)`
    int ret_val = HexagonUserDMA::Get().Copy(&cast_dst[i], &cast_src[i], cur_len);
    if (ret_val != DMA_SUCCESS) return ret_val;
    HexagonUserDMA::Get().Wait(0);
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
