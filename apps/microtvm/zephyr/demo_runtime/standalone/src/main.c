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

/*
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This is a sample Zephyr-based application that contains the logic
 * needed to run a microTVM-based model in standalone mode. This is only
 * intended to be a demonstration, since typically you will want to incorporate
 * this logic into your own application.
 */

#include <drivers/gpio.h>
#include <drivers/uart.h>
#include <fatal.h>
#include <float.h>
#include <kernel.h>
#include <power/reboot.h>
#include <stdio.h>
#include <sys/printk.h>
#include <sys/ring_buffer.h>
#include <tvm/runtime/crt/logging.h>
#include <unistd.h>
#include <zephyr.h>

#include "zephyr_runtime.h"

// ###########################################################################
// UART
// ###########################################################################
static const struct device* g_tvm_uart;
// Ring buffer used to store data read from the UART on rx interrupt.
#define RING_BUF_SIZE_BYTES 4 * 1024
RING_BUF_DECLARE(uart_rx_rbuf, RING_BUF_SIZE_BYTES);

size_t write_serial(const char* data, size_t size) {
  for (size_t i = 0; i < size; i++) {
    uart_poll_out(g_tvm_uart, data[i]);
  }
  return size;
}

static uint8_t uart_data[8];
// UART interrupt callback.
void uart_irq_cb(const struct device* dev, void* user_data) {
  while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
    struct ring_buf* rbuf = (struct ring_buf*)user_data;
    if (uart_irq_rx_ready(dev) != 0) {
      for (;;) {
        // Read a small chunk of data from the UART.
        int bytes_read = uart_fifo_read(dev, uart_data, sizeof(uart_data));
        if (bytes_read < 0) {
          // TVMPlatformAbort((tvm_crt_error_t)(0xbeef1));
        } else if (bytes_read == 0) {
          break;
        }
        // Write it into the ring buffer.
        int bytes_written = ring_buf_put(rbuf, uart_data, bytes_read);
        if (bytes_read != bytes_written) {
          // TVMPlatformAbort((tvm_crt_error_t)(0xbeef2));
        }
        // CHECK_EQ(bytes_read, bytes_written, "bytes_read: %d; bytes_written: %d", bytes_read,
        //         bytes_written);
      }
    }
  }
}

// Used to initialize the UART receiver.
void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);
  uart_irq_rx_enable(dev);
}

// Used to read data from the UART.
int uart_rx_buf_read(struct ring_buf* rbuf, uint8_t* data, size_t data_size_bytes) {
  unsigned int key = irq_lock();
  int bytes_read = ring_buf_get(rbuf, data, data_size_bytes);
  irq_unlock(key);
  return bytes_read;
}

// ###########################################################################
// TVM Model
// ###########################################################################
extern const char graph_json[];
extern unsigned int graph_json_len;
extern const char params_bin[];
extern unsigned int params_bin_len;
extern const char input_bin[];
extern const char output_bin[];

void main(void) {
  // Claim console device.
  g_tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  uart_rx_init(&uart_rx_rbuf, g_tvm_uart);

  TVMLogf("microTVM Zephyr Standalone Demo\n");

  // Create runtime
  int* tvm_handle;
  char* json_data = (char*)(graph_json);
  char* params_data = (char*)(params_bin);
  uint64_t params_size = params_bin_len;
  tvm_handle = tvm_runtime_create(json_data, params_data, params_size);

  // Prepare input/output tensors
  float* input_storage = (float*)input_bin;
  DLTensor input_tensor;
  input_tensor.data = input_storage;
  DLDevice dev = {kDLCPU, 0};
  input_tensor.device = dev;
  input_tensor.ndim = 4;
  DLDataType dtype = {kDLFloat, 32, 1};
  input_tensor.dtype = dtype;
  int64_t in_shape[4] = {1, 1, 28, 28};
  input_tensor.shape = in_shape;
  input_tensor.strides = NULL;
  input_tensor.byte_offset = 0;

  float output_storage[1 * 10];
  DLTensor output_tensor;
  output_tensor.data = output_storage;
  DLDevice out_dev = {kDLCPU, 0};
  output_tensor.device = out_dev;
  output_tensor.ndim = 2;
  DLDataType out_dtype = {kDLFloat, 32, 1};
  output_tensor.dtype = out_dtype;
  int64_t out_shape[2] = {1, 10};
  output_tensor.shape = out_shape;
  output_tensor.strides = NULL;
  output_tensor.byte_offset = 0;

  // Set input tensor
  tvm_runtime_set_input(tvm_handle, "Input3", &input_tensor);

  // Run model
  tvm_runtime_run(tvm_handle);

  // Get output tensor
  tvm_runtime_get_output(tvm_handle, 0, &output_tensor);

  // Destroy the runtime
  tvm_runtime_destroy(tvm_handle);

  int label = -1;
  float max_output = -FLT_MAX;
  for (size_t i = 0; i < (1 * 10); i++) {
    if (output_storage[i] > max_output) {
      max_output = output_storage[i];
      label = i;
    }
  }

  TVMLogf("\nResult: {%d}\n", label);
  TVMPlatformAbort(kTvmErrorNoError);
}
