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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_POWER_MANAGER_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_POWER_MANAGER_H_

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonPowerManager {
 public:
  //! \brief Constructor.
  HexagonPowerManager();

  //! \brief Destructor.
  ~HexagonPowerManager();

  //! \brief Prevent copy construction of HexagonPowerManager.
  HexagonPowerManager(const HexagonPowerManager&) = delete;

  //! \brief Prevent copy assignment with HexagonPowerManager.
  HexagonPowerManager& operator=(const HexagonPowerManager&) = delete;

  //! \brief Prevent move construction.
  HexagonPowerManager(HexagonPowerManager&&) = delete;

  //! \brief Prevent move assignment.
  HexagonPowerManager& operator=(HexagonPowerManager&&) = delete;

 private:
  //! \brief Power context
  void* hap_pwr_ctx_;

  void PowerOnHVX();
  void PowerOffHVX();
  void PowerOnHTP();
  void PowerOffHTP();
  void SetAppType();
  void SetDCVS();
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_POWER_MANAGER_H_
