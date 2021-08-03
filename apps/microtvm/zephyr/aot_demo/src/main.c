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
#include <tvm/runtime/crt/internal/aot_executor/aot_executor.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <unistd.h>
#include <zephyr.h>

#include "input_data.h"
#include "output_data.h"
#include "tvmgen_default.h"
#include "zephyr_uart.h"

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

#define WORKSPACE_SIZE (128 * 1024)

static uint8_t g_aot_memory[WORKSPACE_SIZE];
extern tvm_model_t tvmgen_default_network;
tvm_workspace_t app_workspace;

// Wakeup sequence used to wake up QEMU on the host.
const unsigned char g_wakeup_sequence[12] = {0xfe, 0xff, 0xfd, 0x03, 0x00, 0x00,
                                             0x00, 0x00, 0x00, 0x02, 0x66, 0x77};
const char g_start_cmd[] = "start\n";

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

static uint8_t main_rx_buf[128];
static uint8_t cmd_buf[128];
static size_t g_cmd_buf_ind;

void main(void) {
  g_cmd_buf_ind = 0;
  memset((char*)cmd_buf, 0, sizeof(cmd_buf));
  TVMPlatformUARTInit();
  k_timer_init(&g_microtvm_timer, NULL, NULL);
  // Wake up host side.
  TVMPlatformWriteSerial(g_wakeup_sequence, sizeof(g_wakeup_sequence));

  // Wait for start command
  while (true) {
    int bytes_read = TVMPlatformUartRxRead(main_rx_buf, sizeof(main_rx_buf));
    if (bytes_read > 0) {
      memcpy((char*)cmd_buf + g_cmd_buf_ind, main_rx_buf, bytes_read);
      g_cmd_buf_ind += bytes_read;
    }
    if (g_cmd_buf_ind >= 6) {
      if (!strcmp((char*)(cmd_buf), g_start_cmd)) {
        break;
      } else {
        memset((char*)cmd_buf, 0, sizeof(cmd_buf));
        g_cmd_buf_ind = 0;
      }
    }
  }
  TVMLogf("Zephyr AOT Runtime\n");

  struct tvmgen_default_inputs inputs = {
      .input_1 = input_data,
  };
  struct tvmgen_default_outputs outputs = {
      .output = output_data,
  };

  StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);

  double elapsed_time = 0;
  TVMPlatformTimerStart();
  int ret_val = tvmgen_default_run(&inputs, &outputs);
  TVMPlatformTimerStop(&elapsed_time);

  if (ret_val != 0) {
    TVMLogf("Error: %d\n", ret_val);
    TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
  }

  size_t max_ind = -1;
  float max_val = -FLT_MAX;
  TVMLogf("#result_data*1000:");
  for (size_t i = 0; i < output_data_len; i++) {
    if (output_data[i] >= max_val) {
      max_ind = i;
      max_val = output_data[i];
      TVMLogf(" %d", (int)(output_data[i] * 1000));
    }
  }
  TVMLogf("\n");
  TVMLogf("#result:%d:%d\n", max_ind, (uint32_t)(elapsed_time * 1000));
#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#endif
}
