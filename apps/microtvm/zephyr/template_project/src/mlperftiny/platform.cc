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
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/stack_allocator.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>

#include "crt_config.h"

// WORKSPACE_SIZE is defined in python
static uint8_t g_aot_memory[WORKSPACE_SIZE];
tvm_workspace_t app_workspace;

__attribute__((weak)) size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes,
                                                      const char* fmt, va_list args) {
  return vsnprintk(out_buf, out_buf_size_bytes, fmt, args);
}

__attribute__((weak)) void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMPlatformAbort: %08x\n", error);
  sys_reboot(SYS_REBOOT_COLD);
  for (;;)
    ;
}

__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev,
                                                                void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

__attribute__((weak)) tvm_crt_error_t TVMPlatformInitialize() {
  StackMemoryManager_Init(&app_workspace, g_aot_memory, WORKSPACE_SIZE);
  return kTvmErrorNoError;
}
