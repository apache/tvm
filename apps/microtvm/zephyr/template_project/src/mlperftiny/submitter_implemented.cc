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

#include "api/submitter_implemented.h"

#include <assert.h>
#include <string.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/platform.h>
#include <unistd.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/ring_buffer.h>

#include "api/internally_implemented.h"
#include "crt_config.h"
#include "tvm/output_data.h"
#include "tvmgen_default.h"

// ###############################################################
// Model
// ###############################################################
#define MODEL_KWS 1
#define MODEL_VWW 2
#define MODEL_AD 3
#define MODEL_IC 4

static void* g_input_data;
#if TARGET_MODEL == MODEL_AD
static uint8_t __aligned(4) g_input_data_buffer_aligned[MAX_DB_INPUT_SIZE];
#endif

// OUT_QUANT_SCALE and OUT_QUANT_ZERO are set in python.
#if TARGET_MODEL == MODEL_AD
float* g_output_data = output_data;
#else
int8_t* g_output_data = output_data;
float g_quant_scale = OUT_QUANT_SCALE;
int8_t g_quant_zero = OUT_QUANT_ZERO;
#endif
size_t g_output_data_len = output_data_len;

// ###############################################################
// GPIO
// ###############################################################
#if EE_CFG_ENERGY_MODE == 1 && NRF_BOARD != 1
// use GPIO PC6 which is on connector CN7 pin 1 on the nucleo_l4r5zi
static const char* g_gpio_device_name = "GPIOC";
static const struct device* g_gpio_dev;
static const gpio_pin_t g_gpio_pin = 6;
#endif

// ###############################################################
// UART
// ###############################################################
#define TVM_UART_DEFAULT_BAUDRATE 115200
static const struct device* g_microtvm_uart;

void UartInit(uint32_t baudrate = TVM_UART_DEFAULT_BAUDRATE) {
  // Claim console device.
  g_microtvm_uart = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
  const struct uart_config config = {.baudrate = baudrate,
                                     .parity = UART_CFG_PARITY_NONE,
                                     .stop_bits = UART_CFG_STOP_BITS_1,
                                     .data_bits = UART_CFG_DATA_BITS_8,
                                     .flow_ctrl = UART_CFG_FLOW_CTRL_NONE};
  uart_configure(g_microtvm_uart, &config);
}

char UartRxRead() {
  unsigned char c;
  int ret = -1;
  while (ret != 0) {
    ret = uart_poll_in(g_microtvm_uart, &c);
  }
  return (char)c;
}

uint32_t UartTxWrite(const char* data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    uart_poll_out(g_microtvm_uart, data[i]);
  }
  return size;
}

// ###############################################################
// TVM
// ###############################################################
#ifdef __cplusplus
extern "C" {
#endif
// TODO(mehrdadh): remove and reuse the CRT
// implementation in src/runtime/crt/common/crt_backend_api.c
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

// TODO(mehrdadh): remove and reuse the CRT
// implementation in src/runtime/crt/common/crt_backend_api.c
int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  tvm_crt_error_t err = kTvmErrorNoError;
  DLDevice dev = {(DLDeviceType)device_type, device_id};
  err = TVMPlatformMemoryFree(ptr, dev);
  CHECK_EQ(err, kTvmErrorNoError, "TVMBackendFreeWorkspace(%d, %d)", device_type, device_id);
  return err;
}

void TVMLogf(const char* msg, ...) {
  char buffer[128];
  int size;
  va_list args;
  va_start(args, msg);
  size = TVMPlatformFormatMessage(buffer, 128, msg, args);
  va_end(args);
  UartTxWrite(buffer, (size_t)size);
}

#ifdef __cplusplus
}  // extern "C"
#endif

void Infer(void* input_ptr) {
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
    .Identity = output_data,
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
    th_printf("Error: %d\n", ret_val);
  }
}

// ###############################################################
// MLPerftiny APIs
// ###############################################################
// Implement this method to prepare for inference and preprocess inputs.
// Modified from source
void th_load_tensor() {
#if TARGET_MODEL == MODEL_KWS
  g_input_data = static_cast<void*>(ee_get_buffer_pointer());
#elif TARGET_MODEL == MODEL_VWW
  // Converting uint8 to int8
  int8_t* temp_int = reinterpret_cast<int8_t*>(ee_get_buffer_pointer());
  for (size_t i = 0; i < MAX_DB_INPUT_SIZE; i++) {
    temp_int[i] -= 128;
  }
  g_input_data = static_cast<void*>(temp_int);
#elif TARGET_MODEL == MODEL_AD
  uint8_t* buffer = ee_get_buffer_pointer();
  memcpy(g_input_data_buffer_aligned, buffer, sizeof(g_input_data_buffer_aligned));
  g_input_data = g_input_data_buffer_aligned;
#elif TARGET_MODEL == MODEL_IC
  uint8_t* temp_uint = reinterpret_cast<uint8_t*>(ee_get_buffer_pointer());
  int8_t* temp_int = reinterpret_cast<int8_t*>(ee_get_buffer_pointer());
  for (size_t i = 0; i < MAX_DB_INPUT_SIZE; i++) {
    if (temp_uint[i] <= 127)
      temp_int[i] = ((int8_t)temp_uint[i]) - 128;
    else
      temp_int[i] = (int8_t)(temp_uint[i] - 128);
  }
  g_input_data = reinterpret_cast<void*>(temp_int);
#else
#error Wrong model
#endif
}

#if TARGET_MODEL == MODEL_AD
// calculate |output - input| for AD model
static float calculate_result() {
  size_t feature_size = g_output_data_len;
  float diffsum = 0;
  float* input_float = reinterpret_cast<float*>(g_input_data);
  float* output_float = reinterpret_cast<float*>(g_output_data);

  for (size_t i = 0; i < feature_size; i++) {
    float diff = output_float[i] - input_float[i];
    diffsum += diff * diff;
  }
  diffsum /= feature_size;

  return diffsum;
}
#endif

// Add to this method to return real inference results.
void th_results() {
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
#if TARGET_MODEL == MODEL_AD
  th_printf("m-results-[%0.3f]\r\n", calculate_result());
#else
  size_t kCategoryCount = g_output_data_len;
  th_printf("m-results-[");
  for (size_t i = 0; i < kCategoryCount; i++) {
    float converted = static_cast<float>(g_quant_scale * (g_output_data[i] - g_quant_zero));
    // float converted = static_cast<float>(g_output_data[i]);
    th_printf("%.3f", converted);
    if (i < (kCategoryCount - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
#endif
}

// Implement this method with the logic to perform one inference cycle.
// Modified from source
void th_infer() { Infer(g_input_data); }

/// \brief optional API.
// Modified from source
void th_final_initialize(void) {}

void th_pre() {}
void th_post() {}

void th_command_ready(char volatile* p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char*)p_command);
}

// th_libc implementations.
int th_strncmp(const char* str1, const char* str2, size_t n) { return strncmp(str1, str2, n); }

char* th_strncpy(char* dest, const char* src, size_t n) { return strncpy(dest, src, n); }

size_t th_strnlen(const char* str, size_t maxlen) { return strlen(str); }

char* th_strcat(char* dest, const char* src) { return strcat(dest, src); }

char* th_strtok(char* str1, const char* sep) { return strtok(str1, sep); }

int th_atoi(const char* str) { return atoi(str); }

void* th_memset(void* b, int c, size_t len) { return memset(b, c, len); }

void* th_memcpy(void* dst, const void* src, size_t n) { return memcpy(dst, src, n); }

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char* format, va_list ap) { return vprintf(format, ap); }

// Modified from source
void th_printf(const char* p_fmt, ...) {
  char buffer[128];
  int size;
  va_list args;
  va_start(args, p_fmt);
  size = TVMPlatformFormatMessage(buffer, 128, p_fmt, args);
  va_end(args);
  UartTxWrite(buffer, (size_t)size);
}

// Modified from source
char th_getchar() { return UartRxRead(); }

// Modified from source
void th_serialport_initialize(void) {
#if EE_CFG_ENERGY_MODE == 1 && NRF_BOARD != 1
  UartInit(9600);
#else
  UartInit();
#endif
}

// Modified from source
void th_timestamp(void) {
#if EE_CFG_ENERGY_MODE == 1 && NRF_BOARD != 1
  /* USER CODE 1 BEGIN */
  /* Step 1. Pull pin low */
  gpio_pin_set(g_gpio_dev, g_gpio_pin, 0);
  /* Step 2. Hold low for at least 1us */
  k_busy_wait(1);
  /* Step 3. Release driver */
  gpio_pin_set(g_gpio_dev, g_gpio_pin, 1);
  /* USER CODE 1 END */
#else
  /* USER CODE 2 BEGIN */
  unsigned long microSeconds = (unsigned long)(k_uptime_get() * 1000LL);
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
#endif
}

// Modified from source
void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
#if EE_CFG_ENERGY_MODE == 1 && NRF_BOARD != 1
  g_gpio_dev = device_get_binding(g_gpio_device_name);
  if (g_gpio_dev == NULL) {
    th_printf("GPIO device init failed\r\n");
    return;
  }

  int ret = gpio_pin_configure(g_gpio_dev, g_gpio_pin, GPIO_OUTPUT_HIGH);
  if (ret < 0) {
    th_printf("GPIO pin configure failed\r\n");
    return;
  }
#endif

  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
