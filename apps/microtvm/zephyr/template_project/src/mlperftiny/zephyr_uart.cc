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

#include "zephyr_uart.h"

#include <tvm/runtime/crt/error_codes.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/sys/ring_buffer.h>

#include "crt_config.h"

static const struct device* g_microtvm_uart;

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

// Initialize the UART receiver.
void uart_rx_init(struct ring_buf* rbuf, const struct device* dev) {
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)rbuf);
  uart_irq_rx_enable(dev);
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
  g_microtvm_uart = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
  const struct uart_config config = {.baudrate = baudrate,
                                     .parity = UART_CFG_PARITY_NONE,
                                     .stop_bits = UART_CFG_STOP_BITS_1,
                                     .data_bits = UART_CFG_DATA_BITS_8,
                                     .flow_ctrl = UART_CFG_FLOW_CTRL_NONE};
  uart_configure(g_microtvm_uart, &config);
}
