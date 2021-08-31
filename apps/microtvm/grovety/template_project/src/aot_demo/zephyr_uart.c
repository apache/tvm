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

#include <drivers/uart.h>
#include <sys/ring_buffer.h>

#include "crt_config.h"

static const struct device* g_microtvm_uart;
#define RING_BUF_SIZE_BYTES (TVM_CRT_MAX_PACKET_SIZE_BYTES + 100)

// Ring buffer used to store data read from the UART on rx interrupt.
RING_BUF_DECLARE(uart_rx_rbuf, RING_BUF_SIZE_BYTES);

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

uint32_t TVMPlatformUartRxRead(uint8_t* data, uint32_t data_size_bytes) {
  unsigned int key = irq_lock();
  uint32_t bytes_read = ring_buf_get(&uart_rx_rbuf, data, data_size_bytes);
  irq_unlock(key);
  return bytes_read;
}

uint32_t TVMPlatformWriteSerial(const char* data, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    uart_poll_out(g_microtvm_uart, data[i]);
  }
  return size;
}

int tvm_printf_cb(char c, void* ctx)
{
    uart_poll_out(g_microtvm_uart, c);
    return (int)(unsigned char)c;
}


void TVMLogf(const char* msg, ...) {
  va_list args;
  va_start(args, msg);
  cbvprintf(tvm_printf_cb, NULL, msg, args);
  va_end(args);
}


// Initialize UART
void TVMPlatformUARTInit() {
  // Claim console device.
  g_microtvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  const struct uart_config config = {.baudrate = 115200,
                                     .parity = UART_CFG_PARITY_NONE,
                                     .stop_bits = UART_CFG_STOP_BITS_1,
                                     .data_bits = UART_CFG_DATA_BITS_8,
                                     .flow_ctrl = UART_CFG_FLOW_CTRL_NONE};
  uart_configure(g_microtvm_uart, &config);
  uart_rx_init(&uart_rx_rbuf, g_microtvm_uart);
}
