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

#include <algorithm>

#include "hexagon_common.h"
#include "hexagon_user_dma_descriptors.h"
#include "hexagon_user_dma_instructions.h"
#include "hexagon_user_dma_registers.h"

namespace tvm {
namespace runtime {
namespace hexagon {

int init_hexagon_user_dma() {
#if defined(__hexagon__)
  // reset DMA engine
  unsigned int status = dmpause() & DM0_STATUS_MASK;
  if (status != DM0_STATUS_IDLE) {
    return DMA_FAILURE;
  }
#endif
  return DMA_SUCCESS;
}

int hexagon_user_dma_1d_sync(void* dst, void* src, uint32_t length) {
#if defined(__hexagon__)
  static int config_dma = init_hexagon_user_dma();
  if (config_dma != DMA_SUCCESS) {
    return DMA_FAILURE;
  }

  uint64_t src64 = reinterpret_cast<uint64_t>(src);
  // source address limited to 32 bits
  if (src64 > DESC_SRC_MASK) {
    return DMA_FAILURE;
  }

  uint64_t dst64 = reinterpret_cast<uint64_t>(dst);
  // destination address limited to 32 bits
  if (dst64 > DESC_DST_MASK) {
    return DMA_FAILURE;
  }

  // length limited to 24 bits
  if (length > DESC_LENGTH_MASK) {
    return DMA_FAILURE;
  }

  uint32_t src32 = src64 & DESC_SRC_MASK;
  uint32_t dst32 = dst64 & DESC_DST_MASK;

  void* dma_desc = nullptr;

#ifdef _WIN32
  dma_desc = _aligned_malloc(DMA_DESC_2D_SIZE, DMA_DESC_2D_SIZE);
#else
  int ret = posix_memalign(&dma_desc, DMA_DESC_2D_SIZE, DMA_DESC_2D_SIZE);
  if (ret) {
    return DMA_FAILURE;
  }
#endif

  if (!dma_desc) {
    return DMA_FAILURE;
  }

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

  dmstart(dma_desc);
  unsigned int status = dmwait() & DM0_STATUS_MASK;
  unsigned int done = dma_desc_get_done(dma_desc);

#ifdef _WIN32
  _aligned_free(dma_desc);
#else
  free(dma_desc);
#endif

  if (status == DM0_STATUS_IDLE && done == DESC_DONE_COMPLETE) {
    return DMA_SUCCESS;
  }
  return DMA_FAILURE;
#else
  memcpy(dst, src, length);
  return DMA_SUCCESS;
#endif
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
