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

#include "tvm/platform.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/sys/ring_buffer.h>

#include "crt_config.h"
#include "dlpack/dlpack.h"
#include "tvm/output_data.h"
#include "tvm/runtime/crt/error_codes.h"
#include "tvmgen_default.h"

// WORKSPACE_SIZE is defined in python
static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

// OUT_QUANT_SCALE and OUT_QUANT_ZERO are set in python.
#if TARGET_MODEL == 3
float* g_output_data = output_data;
#else
int8_t* g_output_data = output_data;
float g_quant_scale = OUT_QUANT_SCALE;
int8_t g_quant_zero = OUT_QUANT_ZERO;
#endif
size_t g_output_data_len = output_data_len;

static const struct device* g_microtvm_uart;

void TVMLogf(const char* msg, ...) {
  char buffer[128];
  int size;
  va_list args;
  va_start(args, msg);
  size = TVMPlatformFormatMessage(buffer, 128, msg, args);
  va_end(args);
  TVMPlatformWriteSerial(buffer, (size_t)size);
}

// Called by TVM when a message needs to be formatted.
__attribute__((weak)) size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes,
                                                      const char* fmt, va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

// Called by TVM when an internal invariant is violated, and execution cannot continue.
__attribute__((weak)) void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMPlatformAbort: %08x\n", error);
  sys_reboot(SYS_REBOOT_COLD);
  for (;;)
    ;
}

// Called by TVM when memory allocation is required.
__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev,
                                                                void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

// Called by TVM to free an allocated memory.
__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

#ifdef __cplusplus
extern "C" {
#endif
void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
  tvm_crt_error_t err = kTvmErrorNoError;
  void* ptr = 0;
  DLDevice dev = {(DLDeviceType)device_type, device_id};
  assert(nbytes > 0);
  err = TVMPlatformMemoryAllocate(nbytes, dev, &ptr);
  CHECK_EQ(err, kTvmErrorNoError,
           "TVMBackendAllocWorkspace(%d, %d, %" PRIu64 ", %d, %d) -> %" PRId32, device_type,
           device_id, nbytes, dtype_code_hint, dtype_bits_hint, err);
  return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  tvm_crt_error_t err = kTvmErrorNoError;
  DLDevice dev = {(DLDeviceType)device_type, device_id};
  err = TVMPlatformMemoryFree(ptr, dev);
  CHECK_EQ(err, kTvmErrorNoError, "TVMBackendFreeWorkspace(%d, %d)", device_type, device_id);
  return err;
}

#ifdef __cplusplus
}  // extern "C"
#endif

// TODO
void timer_expiry_function(struct k_timer* timer_id) { return; }

void TVMRuntimeInit() { StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE); }

void TVMInfer(void* input_ptr) {
  struct tvmgen_default_inputs inputs = {
#if TARGET_MODEL == MODEL_KWS
    .input_1 = input_ptr,
#elif TARGET_MODEL == MODEL_IC
    .input_1_int8 = input_ptr,
#elif TARGET_MODEL == MODEL_VWW
    .input_1_int8 = input_ptr,
#elif TARGET_MODEL == MODEL_AD
    .input_1 = input_ptr,
#elif
#error Wrong model.
#endif
  };

  struct tvmgen_default_outputs outputs = {
#if TARGET_MODEL == MODEL_KWS
#if COMPILE_WITH_CMSISNN
    .Identity = output_data,
#else
    .output = output_data,
#endif
#elif TARGET_MODEL == MODEL_IC
    .Identity_int8 = output_data,
#elif TARGET_MODEL == MODEL_VWW
    .Identity_int8 = output_data,
#elif TARGET_MODEL == MODEL_AD
    .Identity = output_data,
#endif
  };

  int ret_val = tvmgen_default_run(&inputs, &outputs);
  if (ret_val != 0) {
    TVMLogf("Error: %d\n", ret_val);
  }
}

int8_t QuantizeFloatToInt8(float value, float scale, int zero_point) {
  int32_t result = round(value / scale) + zero_point;
  if (result < INT8_MIN) {
    result = INT8_MIN;
  }
  if (result > INT8_MAX) {
    result = INT8_MAX;
  }
  return (int8_t)(result);
}

// UART read.
char TVMPlatformUartRxRead() {
  unsigned char c;
  int ret = -1;
  while (ret != 0) {
    ret = uart_poll_in(g_microtvm_uart, &c);
  }
  return (char)c;
}

// UART write.
uint32_t TVMPlatformWriteSerial(const char* data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    uart_poll_out(g_microtvm_uart, data[i]);
  }
  return size;
}

// Initialize UART.
void TVMPlatformUARTInit(uint32_t baudrate /* = TVM_UART_DEFAULT_BAUDRATE */) {
  // Claim console device.
  g_microtvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  const struct uart_config config = {.baudrate = baudrate,
                                     .parity = UART_CFG_PARITY_NONE,
                                     .stop_bits = UART_CFG_STOP_BITS_1,
                                     .data_bits = UART_CFG_DATA_BITS_8,
                                     .flow_ctrl = UART_CFG_FLOW_CTRL_NONE};
  uart_configure(g_microtvm_uart, &config);
}
