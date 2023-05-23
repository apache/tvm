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

/*!
 * \brief Implementation of TVMPlatform functions in tvm/runtime/crt/platform.h
 */

#include <dlpack/dlpack.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/crt/error_codes.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/random/rand32.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/timing/timing.h>

K_HEAP_DEFINE(tvm_heap, TVM_WORKSPACE_SIZE_BYTES);

volatile timing_t g_microtvm_start_time, g_microtvm_end_time;
int g_microtvm_timer_running = 0;

#ifdef CONFIG_LED
#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led0 = GPIO_DT_SPEC_GET(LED0_NODE, gpios);
#endif  // CONFIG_LED

// This is invoked by Zephyr from an exception handler, which will be invoked
// if the device crashes. Here, we turn on the LED and spin.
void k_sys_fatal_error_handler(unsigned int reason, const z_arch_esf_t* esf) {
#ifdef CONFIG_LED
  gpio_pin_set_dt(&led0, 1);
#endif
  for (;;)
    ;
}

void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMError: 0x%x", error);
  sys_reboot(SYS_REBOOT_COLD);
#ifdef CONFIG_LED
  gpio_pin_set_dt(&led0, 1);
#endif
  for (;;)
    ;
}

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  *out_ptr = k_heap_alloc(&tvm_heap, num_bytes, K_NO_WAIT);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  k_heap_free(&tvm_heap, ptr);
  return kTvmErrorNoError;
}

// Called to start system timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_microtvm_timer_running) {
    TVMLogf("timer already running");
    return kTvmErrorPlatformTimerBadState;
  }

#ifdef CONFIG_LED
  gpio_pin_set_dt(&led0, 1);
#endif
  g_microtvm_start_time = timing_counter_get();
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

// Called to stop system timer.
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_microtvm_timer_running) {
    TVMLogf("timer not running");
    return kTvmErrorSystemErrorMask | 2;
  }

#ifdef CONFIG_LED
  gpio_pin_set_dt(&led0, 0);
#endif

  g_microtvm_end_time = timing_counter_get();
  uint64_t cycles = timing_cycles_get(&g_microtvm_start_time, &g_microtvm_end_time);
  uint64_t ns_spent = timing_cycles_to_ns(cycles);
  *elapsed_time_seconds = ns_spent / (double)1e9;
  g_microtvm_timer_running = 0;
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

tvm_crt_error_t TVMPlatformInitialize() {
#ifdef CONFIG_LED
  if (!device_is_ready(led0.port)) {
    for (;;)
      ;
  }
  int ret = gpio_pin_configure_dt(&led0, GPIO_OUTPUT_ACTIVE);
  if (ret < 0) {
    TVMPlatformAbort((tvm_crt_error_t)0xbeef4);
  }
  gpio_pin_set_dt(&led0, 0);
#endif

  // Initialize system timing. We could stop and start it every time, but we'll
  // be using it enough we should just keep it enabled.
  timing_init();
  timing_start();

  return kTvmErrorNoError;
}
