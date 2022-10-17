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
extern "C" {
#include <AEEStdDef.h>
#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_power.h>
#include <qurt_error.h>
}

#include "hexagon_common.h"
#include "hexagon_hmx.h"

// Minimum timeout per SDK docs, excluding 0
#define COMPUTE_RES_ACQ_TIMEOUT 200

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonHmx::HexagonHmx() {
  PowerOn();
  Acquire();
}

HexagonHmx::~HexagonHmx() {
  Release();
  PowerOff();
}

void HexagonHmx::PowerOn() {
  HAP_power_request_t pwr_req;
  int nErr;

  hap_pwr_ctx_ = HAP_utils_create_context();
  pwr_req.type = HAP_power_set_HMX;
  pwr_req.hmx.power_up = true;
  if ((nErr = HAP_power_set(hap_pwr_ctx_, &pwr_req))) {
    LOG(FATAL) << "InternalError: HAP_power_set failed\n";
  }
}

void HexagonHmx::PowerOff() {
  HAP_power_request_t pwr_req;
  int nErr;

  pwr_req.type = HAP_power_set_HMX;
  pwr_req.hmx.power_up = false;
  if ((nErr = HAP_power_set(hap_pwr_ctx_, &pwr_req))) {
    LOG(FATAL) << "InternalError: HAP_power_set failed\n";
  }
  HAP_utils_destroy_context(hap_pwr_ctx_);
}

void HexagonHmx::Acquire() {
  compute_res_attr_t compute_res_attr;
  int nErr;

  if ((nErr = HAP_compute_res_attr_init(&compute_res_attr))) {
    LOG(FATAL) << "InternalError: HAP_compute_res_attr_init failed\n";
  }
  if ((nErr = HAP_compute_res_attr_set_hmx_param(&compute_res_attr, 1))) {
    LOG(FATAL) << "InternalError: HAP_compute_res_attr_set_hmx_param failed\n";
  }
  context_id_ = HAP_compute_res_acquire(&compute_res_attr, COMPUTE_RES_ACQ_TIMEOUT);

  if (!context_id_) {
    LOG(FATAL) << "InternalError: HAP_compute_res_acquire failed\n";
  }
  if ((nErr = HAP_compute_res_hmx_lock(context_id_))) {
    LOG(FATAL) << "InternalError: Unable to lock HMX!";
  }
}

void HexagonHmx::Release() {
  HAP_compute_res_hmx_unlock((unsigned int)context_id_);
  HAP_compute_res_release((unsigned int)context_id_);
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
