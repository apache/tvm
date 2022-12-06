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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_H_

#include "hexagon_common.h"
#include "hexagon_user_dma_descriptors.h"
#include "hexagon_user_dma_instructions.h"
#include "hexagon_user_dma_registers.h"
#include "ring_buffer.h"

namespace tvm {
namespace runtime {
namespace hexagon {

#define DMA_SUCCESS 0
#define DMA_FAILURE -1
#define DMA_RETRY 1
#define MAX_DMA_DESCRIPTORS 100
#define SYNC_DMA_QUEUE -1

class HexagonUserDMA {
 public:
  HexagonUserDMA();
  ~HexagonUserDMA();
  HexagonUserDMA(const HexagonUserDMA&) = delete;
  HexagonUserDMA& operator=(const HexagonUserDMA&) = delete;
  HexagonUserDMA(HexagonUserDMA&&) = delete;
  HexagonUserDMA& operator=(HexagonUserDMA&&) = delete;

  /*!
   * \brief Initiate DMA to copy memory from source to destination address
   * \param dst Destination address
   * \param src Source address
   * \param length Length in bytes to copy
   * \returns Status: DMA_SUCCESS or DMA_FAILURE
   */
  int Copy(int queue_id, void* dst, void* src, uint32_t length, bool bypass_cache);

  /*!
   * \brief Wait until the number of DMAs in flight is less than or equal to some maximum
   * \param max_dmas_in_flight Maximum number of DMAs allowed to be in flight
   * to satisfy the `Wait` e.g. use `Wait(0)` to wait on "all" outstanding DMAs to complete
   */
  void Wait(int queue_id, uint32_t max_dmas_in_flight);

  /*!
   * \brief Poll the number of DMAs in flight
   * \returns Number of DMAs in flight
   */
  uint32_t Poll(int queue_id);

 private:
  //! \brief Initializes the Hexagon User DMA engine
  unsigned int Init();

  //! \brief Calculates and returns the number of DMAs in flight
  uint32_t DMAsInFlight(int queue_id);

  //! \brief Tracks whether the very first DMA has been executed
  bool first_dma_ = true;

  //! \brief Tracks the tail DMA descriptor
  void* tail_dma_desc_ = nullptr;

  //! \brief Storage for all DMA descriptors
  QueuedRingBuffer<dma_desc_2d_t>* descriptors_ = nullptr;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_H_
