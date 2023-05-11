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
#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <unistd.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/ring_buffer.h>

#include "tvm/input_data.h"
#include "tvm/output_data.h"
#include "tvmgen_default.h"

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

// Transport Commands.
// Commands on host end with `\n`
// Commands on microTVM device end with `%`
const unsigned char CMD_WAKEUP[] = "wakeup\n";
const unsigned char CMD_READY[] = "ready\n";
const unsigned char CMD_INIT[] = "init";
const unsigned char CMD_INFER[] = "infer";

#define CMD_SIZE 80u
#define CMD_TERMINATOR '%'

static uint8_t main_rx_buf[128];
static uint8_t g_cmd_buf[128];
static size_t g_cmd_buf_ind;

static const struct device* g_microtvm_uart;
#define RING_BUF_SIZE_BYTES (TVM_CRT_MAX_PACKET_SIZE_BYTES + 100)

// Ring buffer used to store data read from the UART on rx interrupt.
RING_BUF_DECLARE(uart_rx_rbuf, RING_BUF_SIZE_BYTES);

uint32_t UartTxWrite(const char* data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    uart_poll_out(g_microtvm_uart, data[i]);
  }
  return size;
}

uint32_t UartRxRead(uint8_t* data, uint32_t data_size_bytes) {
  unsigned int key = irq_lock();
  uint32_t bytes_read = ring_buf_get(&uart_rx_rbuf, data, data_size_bytes);
  irq_unlock(key);
  return bytes_read;
}

// Initialize UART
void UartInit() {
  // Claim console device.
  g_microtvm_uart = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
  const struct uart_config config = {.baudrate = 115200,
                                     .parity = UART_CFG_PARITY_NONE,
                                     .stop_bits = UART_CFG_STOP_BITS_1,
                                     .data_bits = UART_CFG_DATA_BITS_8,
                                     .flow_ctrl = UART_CFG_FLOW_CTRL_NONE};
  uart_configure(g_microtvm_uart, &config);
  uart_rx_init(&uart_rx_rbuf, g_microtvm_uart);
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
          TVMPlatformAbort((tvm_crt_error_t)(0xbeef1));
        } else if (bytes_read == 0) {
          break;
        }
        // Write it into the ring buffer.
        int bytes_written = ring_buf_put(rbuf, uart_data, bytes_read);
        if (bytes_read != bytes_written) {
          TVMPlatformAbort((tvm_crt_error_t)(0xbeef2));
        }
      }
    }
  }
}

// Used to initialize the UART receiver.
void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);
  uart_irq_rx_enable(dev);
}

void TVMLogf(const char* msg, ...) {
  char buffer[256];
  int size;
  va_list args;
  va_start(args, msg);
  size = vsprintf(buffer, msg, args);
  va_end(args);
  UartTxWrite(buffer, (uint32_t)size);
}

void Infer() {
  struct tvmgen_default_inputs inputs = {
      .input_1 = input_data,
  };
  struct tvmgen_default_outputs outputs = {
      .Identity = output_data,
  };

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
  for (size_t i = 0; i < output_data_len; i++) {
    if (output_data[i] >= max_val) {
      max_ind = i;
      max_val = output_data[i];
    }
  }
  TVMLogf("result:%d:%d\n", max_ind, (uint32_t)(elapsed_time * 1000));
}

// Execute functions based on received command
void command_ready(char* command) {
  if (strncmp(command, CMD_INIT, CMD_SIZE) == 0) {
    UartTxWrite(CMD_WAKEUP, sizeof(CMD_WAKEUP));
  } else if (strncmp(command, CMD_INFER, CMD_SIZE) == 0) {
    Infer();
  } else {
    UartTxWrite(CMD_READY, sizeof(CMD_READY));
  }
}

// Append received characters to buffer and check for termination character.
void serial_callback(char* message, int len_bytes) {
  for (int i = 0; i < len_bytes; i++) {
    if (message[i] == CMD_TERMINATOR) {
      g_cmd_buf[g_cmd_buf_ind] = (char)0;
      command_ready(g_cmd_buf);
      g_cmd_buf_ind = 0;
    } else {
      g_cmd_buf[g_cmd_buf_ind] = message[i];
      g_cmd_buf_ind += 1;
    }
  }
}

void main(void) {
  TVMPlatformInitialize();
  UartInit();
  g_cmd_buf_ind = 0;
  memset((char*)g_cmd_buf, 0, sizeof(g_cmd_buf));

  while (true) {
    int bytes_read = UartRxRead(main_rx_buf, sizeof(main_rx_buf));
    if (bytes_read > 0) {
      serial_callback(main_rx_buf, bytes_read);
    }
  }

#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#endif
}
