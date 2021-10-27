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

#include "tvm_hvx.h"

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_power.h"

extern "C" {
#include "qurt_error.h"
#include "qurt_hvx.h"
}

namespace hvx {

#if __HEXAGON_ARCH__ >= 65
#define DEFAULT_HVX_MODE MODE_128B
#else
#define DEFAULT_HVX_MODE MODE_DONT_CARE
#endif

static constexpr mode_t default_hvx_mode = DEFAULT_HVX_MODE;

int reserve(unsigned num_units) {
  if (qurt_hvx_get_units() <= 0) {
    return -1;  // HVX not supported in this target.
  }

  if (num_units == 0) num_units = QURT_HVX_RESERVE_ALL_AVAILABLE;
  int ret_val = qurt_hvx_reserve(num_units);
  switch (ret_val) {
    case QURT_HVX_RESERVE_ALREADY_MADE:
    case QURT_HVX_RESERVE_NOT_SUPPORTED:
    case QURT_HVX_RESERVE_NOT_SUCCESSFUL:
      return 0;

    default:
      if (ret_val < 0) {
        return -1;
      }
      break;
  }
  return ret_val;
}

int unreserve() {
  int ret_val = qurt_hvx_cancel_reserve();
  if (ret_val != QURT_EOK) {
    return -1;
  }
  return 0;
}

int power_on() {
  HAP_power_request_t request;
  request.type = HAP_power_set_HVX;
  request.hvx.power_up = 1;
  int rc = HAP_power_set(nullptr, &request);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "%s: unable to power on HVX, rc=%08x", rc);
    return -1;
  }
  return 0;
}

int power_off() {
  HAP_power_request_t request;
  request.type = HAP_power_set_HVX;
  request.hvx.power_up = 0;
  int rc = HAP_power_set(nullptr, &request);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "%s: unable to power off HVX, rc=%08x", rc);
    return -1;
  }
  return 0;
}

int lock(mode_t mode) {
  qurt_hvx_mode_t qurt_mode;
  int vlen;

  if (MODE_DONT_CARE == mode) mode = default_hvx_mode;

  switch (mode) {
    case MODE_DONT_CARE: {
      int ret_val = qurt_hvx_get_mode();
      if (ret_val < 0) {
        FARF(HIGH, "%s: unknown HVX mode %d", __func__, qurt_mode);
        return -1;
      }
      qurt_mode = static_cast<qurt_hvx_mode_t>(ret_val);
      switch (qurt_mode) {
        case QURT_HVX_MODE_64B:
          vlen = 64;
          break;
        case QURT_HVX_MODE_128B:
          vlen = 128;
          break;
      }
      break;
    }

    case MODE_64B:
      qurt_mode = QURT_HVX_MODE_64B;
      vlen = 64;
      break;

    case MODE_128B:
      qurt_mode = QURT_HVX_MODE_128B;
      vlen = 128;
      break;

    default:
      FARF(HIGH, "%s: unknown HVX mode %d", __func__, qurt_mode);
      return -3;
  }

  // Starting with v65, the RTOS supports HVX context switching.
  // Treat all hvx locks as blocking now, so they can succeed, and
  // be scheduled according to RTOS scheduler via thread priority.
  // Nonblocking call: qurt_hvx_try_lock(qurt_mode).
  int ret_val = qurt_hvx_lock(qurt_mode);

  if (ret_val != QURT_EOK) {
    return -1;
  }
  return vlen;
}

int unlock() {
  int ret_val = qurt_hvx_unlock();
  if (ret_val != QURT_EOK) {
    return -1;
  }
  return 0;
}

int prepare_mt_job(config_t* hvx_config) {
  int num_units = qurt_hvx_get_units();
  if (num_units <= 0) {
    return -1;
  }

  // Check whether HVX is reserved for this protection domain. If not,
  // see if we can temporarily reserve them for this invocation only.
  hvx_config->temp_reserve = false;
  if (hvx_config->num_reserved == 0) {
    hvx_config->num_reserved = reserve(0);  // Reserve all units.
    if (hvx_config->num_reserved <= 0) {
      return -1;
    }
    hvx_config->temp_reserve = true;
  }

  // If client doesn't specify required mode, fallback to default.
  if (hvx_config->mode == MODE_DONT_CARE) hvx_config->mode = default_hvx_mode;

  // Choose 64 byte or 128 byte mode, based on whether there are odd or even
  // number of units
  if (hvx_config->mode == MODE_64B ||
      (hvx_config->mode == MODE_DONT_CARE && (hvx_config->num_reserved & 1))) {
    hvx_config->vlen = 64;
    hvx_config->mode = MODE_64B;
    hvx_config->num_threads = hvx_config->num_reserved;
  } else {
    hvx_config->vlen = 128;
    hvx_config->mode = MODE_128B;
    hvx_config->num_threads = (num_units >> 8) & 0xFF;
    // Handle case where only 1 64-byte unit was available.
    if (hvx_config->num_threads == 0) {
      if (hvx_config->temp_reserve) unreserve();
      return -1;
    }
  }

  // If using HVX, make sure it turns on properly.
  if (hvx_config->num_reserved > 0 && power_on() != 0) {
    return -1;
  }
  return 0;
}

int cleanup_mt_job(const config_t* hvx_config) {
  // If HVX was used, indicate it can be turned off.
  if (hvx_config->num_reserved > 0) power_off();
  // If HVX was temporarily reserved, unreserve it.
  if (hvx_config->temp_reserve) unreserve();
  return 0;
}

}  // namespace hvx
