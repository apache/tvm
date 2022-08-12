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

#include <vector>

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_H_

namespace tvm {
namespace runtime {
namespace hexagon {

#define DMA_SUCCESS 0
#define DMA_FAILURE -1

class HexagonUserDMA {
 public:
  int Copy(void* dst, void* src, uint32_t length);
  void Wait(uint32_t max_dmas_in_flight);
  uint32_t Poll();

  static HexagonUserDMA& Get() {
    static HexagonUserDMA* hud = new HexagonUserDMA();
    return *hud;
  }

 private:
  HexagonUserDMA() = default;
  ~HexagonUserDMA();
  HexagonUserDMA(const HexagonUserDMA&) = delete;
  HexagonUserDMA& operator=(const HexagonUserDMA&) = delete;
  HexagonUserDMA(HexagonUserDMA&&) = delete;
  HexagonUserDMA& operator=(HexagonUserDMA&&) = delete;

  unsigned int Init();
  uint32_t DMAsInFlight();

  bool first_dma_{true};
  uint32_t oldest_dma_in_flight_{0};
  std::vector<void*> dma_descriptors_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_H_
