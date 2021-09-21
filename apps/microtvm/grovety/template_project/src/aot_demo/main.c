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
#include <zephyr.h>

#include "model_data.h"
#include "tvmgen_default.h"
#include "zephyr_uart.h"
#include "perf_timer.h"

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

#define WORKSPACE_SIZE (174 * 1024)

static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

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

static uint8_t cmd_buf[128];
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

  if (end != NULL)
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

  while (true) {
    int readed = TVMPlatformUartRxRead(ptr, 1);
    if (readed <= 0) continue;
    if (*ptr == '\n' || *ptr == ':') {
      *ptr = 0;
      break;
    }
    ptr++;
  }

  if (strncmp((char*)(cmd_buf), "#input", 6))
    return -1;

  // TVMLogf("input readed\n");
  ptr = cmd_buf;
  int index = 0;
  while (index < INPUT_DATA_LEN) {
    int readed = TVMPlatformUartRxRead(ptr, 1);
    if (readed > 0) {
      if (*ptr == ',') {
        *ptr = 0;
        input_data[index] = _strtof(cmd_buf, NULL);
        ptr = cmd_buf;
        index++;
      } else if (*ptr == '\n') {
        *ptr = 0;
        input_data[index] = _strtof(cmd_buf, NULL);
        break;
      } else {
        ptr++;
      }
    }
  }

  return 0;
}

void print_result() {
  TVMLogf("#result:");
  for (int i = 0; i < OUTPUT_DATA_LEN; i++) {
    float y = output_data[i];
    TVMLogf("%.3f", y);
    if (i < OUTPUT_DATA_LEN - 1) TVMLogf(",");
  }
  uint32_t elapsed_time_us = 0;

  for (int i = 0; i < PERF_TIMER_NUMBER_OPS; i++) {
    elapsed_time_us = perf_timer_get_counter(i) / 1000;
    TVMLogf(":%d", elapsed_time_us);
  }

  TVMLogf("\n");
}

void main(void) {
  TVMPlatformUARTInit();

  TVMLogf("Zephyr AOT Runtime\n");

  struct tvmgen_default_inputs inputs = {
      .model_input_0 = input_data,
  };
  struct tvmgen_default_outputs outputs = {
      .output = output_data,
  };

  StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);

  for (int index = 0;; index++) {
    input_data[0] = get_input();

    perf_timer_clear_all();
    perf_timer_start(PERF_TIMER_TOTAL);

    int ret_val = tvmgen_default_run(&inputs, &outputs);

    perf_timer_stop(PERF_TIMER_TOTAL);

    if (ret_val != 0) {
      TVMLogf("Error: %d\n", ret_val);
      TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
    }

    print_result();
  }
#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#endif
}
