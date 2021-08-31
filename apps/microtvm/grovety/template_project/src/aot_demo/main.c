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

#include <assert.h>
#include <float.h>
#include <kernel.h>
#include <power/reboot.h>
#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <unistd.h>
#include <zephyr.h>

#include "model_data.h"
#include "tvmgen_default.h"
#include "zephyr_uart.h"

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

#define WORKSPACE_SIZE (200 * 1024)

static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

void TVMLogf(const char* msg, ...) {
  char buffer[256];
  int size;
  va_list args;
  va_start(args, msg);
  size = vsprintf(buffer, msg, args);
  va_end(args);
  TVMPlatformWriteSerial(buffer, (uint32_t)size);
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

void timer_expiry_function(struct k_timer* timer_id) { return; }

#define MILLIS_TIL_EXPIRY 200
#define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
struct k_timer g_microtvm_timer;
uint32_t g_microtvm_start_time;
int g_microtvm_timer_running = 0;

// Called to start system timer.
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

// Called to stop system timer.
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

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  tvm_crt_error_t err = kTvmErrorNoError;
  void* ptr = 0;
  DLDevice dev = {device_type, device_id};
  assert(nbytes > 0);
  err = TVMPlatformMemoryAllocate(nbytes, dev, &ptr);
  CHECK_EQ(err, kTvmErrorNoError,
           "TVMBackendAllocWorkspace(%d, %d, %" PRIu64 ", %d, %d) -> %" PRId32, device_type,
           device_id, nbytes, dtype_code_hint, dtype_bits_hint, err);
  return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  tvm_crt_error_t err = kTvmErrorNoError;
  DLDevice dev = {device_type, device_id};
  err = TVMPlatformMemoryFree(ptr, dev);
  return err;
}

static uint8_t cmd_buf[512];
#include <ctype.h>

float _strtof(const char* start, const char** end) {
  float f_part = 0.0f;
  float i_part = 0.0f;
  float sign = 1;

  const char* ptr = start;
  while (*ptr && !isdigit(*ptr)) ptr++;

  if (isdigit(*ptr) && ptr > start && *(ptr - 1) == '-') sign = -1;

  while (*ptr && isdigit(*ptr)) {
    i_part = i_part * 10 + (*ptr - '0');
    ptr++;
  }

  if (*ptr == '.') {
    ptr++;
    int d = 10;

    while (*ptr && isdigit(*ptr)) {
      f_part += (float)(*ptr - '0') / d;
      ptr++;
      d *= 10;
    }
  }

  *end = ptr;
  return sign * (i_part + f_part);
}

// Wait for input command
float get_input() {
  char* ptr = cmd_buf;

  // waiting for statrting '#' symbol
  while (true) {
    int readed = TVMPlatformUartRxRead(ptr, 1);
    if (readed > 0 && *ptr == '#') {
      ptr++;
      break;
    }
  }

  char* str_end = NULL;
  // waiting for EOL
  while (true) {
    int readed = TVMPlatformUartRxRead(ptr, 1);
    if (readed > 0) {
      if (*ptr != '\n') {
        ptr++;
      } else {
        *ptr = 0;
        str_end = ptr;
        break;
      }
    }
  }

  TVMLogf("Readed str: %s\n", cmd_buf);

  if (!strncmp((char*)(cmd_buf), "#input:", 7)) {
    ptr = cmd_buf + 7;
    char* end = NULL;
    // float val = strtof(ptr, &end);
    // float val = (float)atof(ptr);
    float val = _strtof(ptr, &end);

    int val1 = (int)(val * 1000);

    TVMLogf("Readed val: %d.%03d  len = %d\n", val1 / 1000, abs(val1 % 1000), (int)(end - ptr));

    return val;
  }
  return 0;
}

void main(void) {
  TVMPlatformUARTInit();
  k_timer_init(&g_microtvm_timer, NULL, NULL);

  TVMLogf("Zephyr AOT Runtime\n");

  struct tvmgen_default_inputs inputs = {
      .model_input_0 = input_data,
  };
  struct tvmgen_default_outputs outputs = {
      .output = output_data,
  };

  StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);

  for (int index = 0;; index++) {
    TVMLogf("Step %3d..... ", index);
    input_data[0] = get_input();
    double elapsed_time = 0;

    TVMPlatformTimerStart();
    int ret_val = tvmgen_default_run(&inputs, &outputs);
    TVMPlatformTimerStop(&elapsed_time);
    TVMLogf("DONE: \n");

    if (ret_val != 0) {
      TVMLogf("Error: %d\n", ret_val);
      TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
    }

    float y = output_data[0];

    int y1 = (int)(y * 1000);
    TVMLogf("#result:%d.%03d:%d\n", y1 / 1000, abs(y1 % 1000), (uint32_t)(elapsed_time * 1000000));

    printf("%f\n", y);
  }
#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#endif
}
