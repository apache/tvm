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

/*!
 * \brief Implementation of TVMPlatform functions in tvm/runtime/crt/platform.h
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#include "crt_config.h"
#include "dlpack/dlpack.h"
#include "tvmgen_default.h"

// TVM_WORKSPACE_SIZE_BYTES defined in Project API Makefile
static uint8_t g_aot_memory[TVM_WORKSPACE_SIZE_BYTES];
tvm_workspace_t app_workspace;

#define MILLIS_TIL_EXPIRY 200
#define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
struct k_timer g_microtvm_timer;
uint32_t g_microtvm_start_time;
int g_microtvm_timer_running = 0;

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMPlatformAbort: %08x\n", error);
  sys_reboot(SYS_REBOOT_COLD);
  for (;;)
    ;
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

tvm_crt_error_t TVMPlatformInitialize() {
  k_timer_init(&g_microtvm_timer, NULL, NULL);
  StackMemoryManager_Init(&app_workspace, g_aot_memory, sizeof(g_aot_memory));
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_microtvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorPlatformTimerBadState;
  }

  k_timer_start(&g_microtvm_timer, TIME_TIL_EXPIRY, TIME_TIL_EXPIRY);
  g_microtvm_start_time = k_cycle_get_32();
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_microtvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorSystemErrorMask | 2;
  }

  uint32_t stop_time = k_cycle_get_32();

  // compute how long the work took
  uint32_t cycles_spent = stop_time - g_microtvm_start_time;
  if (stop_time < g_microtvm_start_time) {
    // we rolled over *at least* once, so correct the rollover it was *only*
    // once, because we might still use this result
    cycles_spent = ~((uint32_t)0) - (g_microtvm_start_time - stop_time);
  }

  uint32_t ns_spent = (uint32_t)k_cyc_to_ns_floor64(cycles_spent);
  double hw_clock_res_us = ns_spent / 1000.0;

  // need to grab time remaining *before* stopping. when stopped, this function
  // always returns 0.
  int32_t time_remaining_ms = k_timer_remaining_get(&g_microtvm_timer);
  k_timer_stop(&g_microtvm_timer);
  // check *after* stopping to prevent extra expiries on the happy path
  if (time_remaining_ms < 0) {
    return kTvmErrorSystemErrorMask | 3;
  }
  uint32_t num_expiries = k_timer_status_get(&g_microtvm_timer);
  uint32_t timer_res_ms = ((num_expiries * MILLIS_TIL_EXPIRY) + time_remaining_ms);
  double approx_num_cycles =
      (double)k_ticks_to_cyc_floor32(1) * (double)k_ms_to_ticks_ceil32(timer_res_ms);
  // if we approach the limits of the HW clock datatype (uint32_t), use the
  // coarse-grained timer result instead
  if (approx_num_cycles > (0.5 * (~((uint32_t)0)))) {
    *elapsed_time_seconds = timer_res_ms / 1000.0;
  } else {
    *elapsed_time_seconds = hw_clock_res_us / 1e6;
  }

  g_microtvm_timer_running = 0;
  return kTvmErrorNoError;
}
