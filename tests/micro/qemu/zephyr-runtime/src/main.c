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

#include <drivers/gpio.h>
#include <drivers/uart.h>
#include <kernel.h>
#include <power/reboot.h>
#include <random/rand32.h>
#include <stdio.h>
#include <sys/printk.h>
#include <sys/ring_buffer.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/utvm_rpc_server.h>
#include <unistd.h>
#include <zephyr.h>

#ifdef CONFIG_ARCH_POSIX
#include "posix_board_if.h"
#endif

#include "crt_config.h"

K_SEM_DEFINE(tx_sem, 0, 1);

static const struct device* tvm_uart;

int write_hook(int c) {
  uart_poll_out(tvm_uart, c);
  return 0;
}

ssize_t write_serial(void* unused_context, const uint8_t* data, size_t size) {
  for (size_t i = 0; i < size; i++) {
    uart_poll_out(tvm_uart, data[i]);
  }

  return size;
}

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

void TVMPlatformAbort(tvm_crt_error_t error) {
  sys_reboot(SYS_REBOOT_COLD);
  for (;;)
    ;
}

K_MEM_POOL_DEFINE(tvm_memory_pool, 64, 1024, 120, 4);

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLContext ctx, void** out_ptr) {
  *out_ptr = k_mem_pool_malloc(&tvm_memory_pool, num_bytes);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLContext ctx) {
  k_free(ptr);
  return kTvmErrorNoError;
}

uint32_t g_utvm_start_time;

#define MILLIS_TIL_EXPIRY 200
#define TIME_TIL_EXPIRY (K_MSEC(MILLIS_TIL_EXPIRY))
K_TIMER_DEFINE(g_utvm_timer, /* expiry func */ NULL, /* stop func */ NULL);

int g_utvm_timer_running = 0;

#ifdef CONFIG_LED
/* The devicetree node identifier for the "led0" alias. */
#define LED0_NODE DT_ALIAS(led0)

#define LED0 DT_GPIO_LABEL(LED0_NODE, gpios)
#define PIN DT_GPIO_PIN(LED0_NODE, gpios)
#define FLAGS DT_GPIO_FLAGS(LED0_NODE, gpios)

static struct device* led_pin;
#endif  // CONFIG_LED

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorPlatformTimerBadState;
  }

#ifdef CONFIG_LED
  gpio_pin_set(led_pin, PIN, 1);
#endif
  k_timer_start(&g_utvm_timer, TIME_TIL_EXPIRY, TIME_TIL_EXPIRY);
  g_utvm_start_time = k_cycle_get_32();
  g_utvm_timer_running = 1;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_utvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorPlatformTimerBadState;
  }

  uint32_t stop_time = k_cycle_get_32();
#ifdef CONFIG_LED
  gpio_pin_set(led_pin, PIN, 0);
#endif

  // compute how long the work took
  uint32_t cycles_spent = stop_time - g_utvm_start_time;
  if (stop_time < g_utvm_start_time) {
    // we rolled over *at least* once, so correct the rollover it was *only*
    // once, because we might still use this result
    cycles_spent = ~((uint32_t)0) - (g_utvm_start_time - stop_time);
  }

  uint32_t ns_spent = (uint32_t)k_cyc_to_ns_floor64(cycles_spent);
  double hw_clock_elapsed_seconds = ns_spent / 1e9;

  // need to grab time remaining *before* stopping. when stopped, this function
  // always returns 0.
  int32_t time_remaining_ms = k_timer_remaining_get(&g_utvm_timer);
  k_timer_stop(&g_utvm_timer);
  // check *after* stopping to prevent extra expiries on the happy path
  if (time_remaining_ms < 0) {
    TVMLogf("negative time remaining");
    return -1;
  }
  uint32_t num_expiries = k_timer_status_get(&g_utvm_timer);
  uint32_t timer_res_ms = ((num_expiries * MILLIS_TIL_EXPIRY) + time_remaining_ms);
  double approx_num_cycles =
      (double)k_ticks_to_cyc_floor32(1) * (double)k_ms_to_ticks_ceil32(timer_res_ms);
  // if we approach the limits of the HW clock datatype (uint32_t), use the
  // coarse-grained timer result instead
  if (approx_num_cycles > (0.5 * (~((uint32_t)0)))) {
    *elapsed_time_seconds = timer_res_ms / 1e3;
  } else {
    *elapsed_time_seconds = hw_clock_elapsed_seconds;
  }

  g_utvm_timer_running = 0;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  uint32_t random;  // one unit of random data.

  // Fill parts of `buffer` which are as large as `random`.
  size_t num_full_blocks = num_bytes / sizeof(random);
  for (int i = 0; i < num_full_blocks; ++i) {
    random = sys_rand32_get();
    memcpy(&buffer[i * sizeof(random)], &random, sizeof(random));
  }

  // Fill any leftover tail which is smaller than `random`.
  size_t num_tail_bytes = num_bytes % sizeof(random);
  if (num_tail_bytes > 0) {
    random = sys_rand32_get();
    memcpy(&buffer[num_bytes - num_tail_bytes], &random, num_tail_bytes);
  }

  return kTvmErrorNoError;
}

#define RING_BUF_SIZE 512
struct uart_rx_buf_t {
  struct ring_buf buf;
  uint32_t buffer[RING_BUF_SIZE];
};

struct uart_rx_buf_t uart_rx_buf;

void uart_irq_cb(const struct device* dev, void* user_data) {
  while (uart_irq_update(dev) && uart_irq_is_pending(dev)) {
    struct uart_rx_buf_t* buf = (struct uart_rx_buf_t*)user_data;
    if (uart_irq_rx_ready(dev) == 0) {
      continue;
    }

    uint8_t data[32];
    for (;;) {
      int bytes_read = uart_fifo_read(dev, data, sizeof(data));
      if (bytes_read < 0) {
        TVMPlatformAbort(0xbeef);
      } else if (bytes_read == 0) {
        break;
      }
      int bytes_written = ring_buf_put(&buf->buf, data, bytes_read);
      CHECK_EQ(bytes_read, bytes_written, "bytes_read: %d; bytes_written: %d", bytes_read,
               bytes_written);
    }
  }
}

void uart_rx_init(struct uart_rx_buf_t* buf, const struct device* dev) {
  ring_buf_init(&buf->buf, RING_BUF_SIZE, buf->buffer);
  uart_irq_callback_user_data_set(dev, uart_irq_cb, (void*)buf);
  uart_irq_rx_enable(dev);
}

int uart_rx_buf_read(struct uart_rx_buf_t* buf, uint8_t* data, size_t data_size_bytes) {
  unsigned int key = irq_lock();
  int bytes_read = ring_buf_get(&buf->buf, data, data_size_bytes);
  irq_unlock(key);
  return bytes_read;
}

extern void __stdout_hook_install(int (*hook)(int));
void main(void) {
#ifdef CONFIG_LED
  led_pin = device_get_binding(LED0);
  if (led_pin == NULL) {
    for (;;)
      ;
  }
  int ret = gpio_pin_configure(led_pin, PIN, GPIO_OUTPUT_ACTIVE | FLAGS);
  if (ret < 0) {
    for (;;)
      ;
  }
  gpio_pin_set(led_pin, PIN, 0);
#endif

  /* Claim console device */
  tvm_uart = device_get_binding(DT_LABEL(DT_CHOSEN(zephyr_console)));
  uart_rx_init(&uart_rx_buf, tvm_uart);
  __stdout_hook_install(&write_hook);

  utvm_rpc_server_t server = UTvmRpcServerInit(write_serial, NULL);
  TVMLogf("uTVM On-Device Runtime");

  while (true) {
    uint8_t buf[256];
    int bytes_read = uart_rx_buf_read(&uart_rx_buf, buf, sizeof(buf));
    if (bytes_read > 0) {
      size_t bytes_remaining = bytes_read;
      uint8_t* cursor = buf;
      while (bytes_remaining > 0) {
        tvm_crt_error_t err = UTvmRpcServerLoop(server, &cursor, &bytes_remaining);
        if (err != kTvmErrorNoError && err != kTvmErrorFramingShortPacket) {
          TVMPlatformAbort(err);
        }
      }
    }
  }

#ifdef CONFIG_ARCH_POSIX
  posix_exit(0);
#endif
}
