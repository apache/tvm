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

#include <stdint.h>

namespace tvm {
namespace runtime {
namespace hexagon {

#define DMA_SUCCESS 0
#define DMA_FAILURE -1

class HexagonUserDMA {
 public:
  /*!
   * \brief Initiate DMA to copy memory from source to destination address
   * \param dst Destination address
   * \param src Source address
   * \param length Length in bytes to copy
   * \returns Status, either DMA_SUCCESS or DMA_FAILURE
   */
  int Copy(void* dst, void* src, uint32_t length);

  /*!
   * \brief Wait until the number of DMAs in flight is less than or equal to some maximum
   * \param max_dmas_in_flight Maximum number of DMAs allowed to be in flight
   * to satisfy the `Wait` e.g. use `Wait(0)` to wait on "all" outstanding DMAs to complete
   */
  void Wait(uint32_t max_dmas_in_flight);

  /*!
   * \brief Poll the number of DMAs in flight
   * \returns Number of DMAs in flight
   */
  uint32_t Poll();

  //! HexagonUserDMA uses the singleton pattern
  static HexagonUserDMA& Get() {
    static HexagonUserDMA* hud = new HexagonUserDMA();
    return *hud;
  }

 private:
  HexagonUserDMA();
  ~HexagonUserDMA();
  HexagonUserDMA(const HexagonUserDMA&) = delete;
  HexagonUserDMA& operator=(const HexagonUserDMA&) = delete;
  HexagonUserDMA(HexagonUserDMA&&) = delete;
  HexagonUserDMA& operator=(HexagonUserDMA&&) = delete;

  //! \brief Initializes the Hexagon User DMA engine
  unsigned int Init();

  //! \brief Calculates and returns the number of DMAs in flight; updates the ID of the oldest
  //! descriptor in flight
  uint32_t DMAsInFlight();

  //! \brief Calculates and returns the address of a DMA descriptor in the ring buffer given a
  //! descriptor ID
  void* GetDescriptorAddr(uint32_t dma_desc_id);

  //! \brief Pointer to ring buffer storage for all DMA descriptors
  void* dma_desc_ring_buff_{nullptr};

  //! \brief Size of ring buffer storage for all DMA descriptors
  const uint32_t dma_desc_ring_buff_size_{100};

  //! \brief Tracks both the total number of DMA descriptors and the ID of the next DMA descriptor
  //! to be added to the ring buffer - modulo ring buffer size to find the ring buffer index for the
  //! next DMA descriptor
  uint32_t id_next_dma_desc_{0};

  //! \brief Tracks the ID of the oldest DMA descriptor in flight OR the ID of the next DMA
  //! descriptor if no DMA descriptors are in flight  - modulo ring buffer size to find the ring
  //! buffer index for the oldest DMA descriptor in flight
  uint32_t id_oldest_dma_desc_in_flight_{0};

  //! \brief Tracks whether (or not) we are executing the very first DMA
  bool first_dma_{true};
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_H_
