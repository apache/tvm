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

#include "hexagon_power_manager.h"

#include <AEEStdDef.h>
#include <AEEStdErr.h>

#include "HAP_power.h"
#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonPowerManager::HexagonPowerManager() {
  hap_pwr_ctx_ = HAP_utils_create_context();
  PowerOnHVX();
  PowerOnHTP();
  SetAppType();
  SetDCVS();
}

HexagonPowerManager::~HexagonPowerManager() {
  PowerOffHTP();
  PowerOffHVX();
  HAP_utils_destroy_context(hap_pwr_ctx_);
}

void HexagonPowerManager::PowerOnHVX() {
  HAP_power_request_t pwr_req;

  pwr_req.type = HAP_power_set_HVX;
  pwr_req.hvx.power_up = true;
  HEXAGON_SAFE_CALL(HAP_power_set(hap_pwr_ctx_, &pwr_req));
}

void HexagonPowerManager::PowerOffHVX() {
  HAP_power_request_t pwr_req;

  pwr_req.type = HAP_power_set_HVX;
  pwr_req.hvx.power_up = false;
  HEXAGON_SAFE_CALL(HAP_power_set(hap_pwr_ctx_, &pwr_req));
}

void HexagonPowerManager::PowerOnHTP() {
  HAP_power_request_t pwr_req;

  pwr_req.type = HAP_power_set_HMX;
  pwr_req.hmx.power_up = true;
  HEXAGON_SAFE_CALL(HAP_power_set(hap_pwr_ctx_, &pwr_req));
}

void HexagonPowerManager::PowerOffHTP() {
  HAP_power_request_t pwr_req;

  pwr_req.type = HAP_power_set_HMX;
  pwr_req.hmx.power_up = false;
  HEXAGON_SAFE_CALL(HAP_power_set(hap_pwr_ctx_, &pwr_req));
}

void HexagonPowerManager::SetAppType() {
  HAP_power_request_t pwr_req;

  pwr_req.type = HAP_power_set_apptype;
  pwr_req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
  HEXAGON_SAFE_CALL(HAP_power_set(hap_pwr_ctx_, &pwr_req));
}

void HexagonPowerManager::SetDCVS() {
  HAP_power_request_t pwr_req;

  memset(&pwr_req, 0, sizeof(HAP_power_request_t));
  pwr_req.type = HAP_power_set_DCVS_v3;
  pwr_req.dcvs_v3.set_dcvs_enable = TRUE;
  pwr_req.dcvs_v3.dcvs_enable = FALSE;
  pwr_req.dcvs_v3.set_core_params = TRUE;
  pwr_req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
  pwr_req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
  pwr_req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
  pwr_req.dcvs_v3.set_bus_params = TRUE;
  pwr_req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
  pwr_req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
  pwr_req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_PLUS;
  pwr_req.dcvs_v3.set_sleep_disable = TRUE;
  pwr_req.dcvs_v3.sleep_disable = TRUE;
  HEXAGON_SAFE_CALL(HAP_power_set(hap_pwr_ctx_, &pwr_req));
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
