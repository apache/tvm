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

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/reboot.h>

#include "dlpack/dlpack.h"
#include "tvm/runtime/crt/error_codes.h"

// Heap for use by TVMPlatformMemoryAllocate.
K_HEAP_DEFINE(tvm_heap, HEAP_SIZE_BYTES);

// Called by TVM when a message needs to be formatted.
__attribute__((weak)) size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes,
                                                      const char* fmt, va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

// Called by TVM when an internal invariant is violated, and execution cannot continue.
__attribute__((weak)) void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMError: 0x%x", error);
  sys_reboot(SYS_REBOOT_COLD);
#ifdef CONFIG_LED
  gpio_pin_set(led0_pin, LED0_PIN, 1);
#endif
  for (;;)
    ;
}

// Called by TVM when memory allocation is required.
__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev,
                                                                void** out_ptr) {
  *out_ptr = k_heap_alloc(&tvm_heap, num_bytes, K_NO_WAIT);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to free an allocated memory.
__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  k_heap_free(&tvm_heap, ptr);
  return kTvmErrorNoError;
}
