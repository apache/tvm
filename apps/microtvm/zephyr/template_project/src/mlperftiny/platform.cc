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

// #ifdef __cplusplus
// extern "C" {
// #endif
// // TODO(mehrdadh): remove and reuse the CRT
// // implementation in src/runtime/crt/common/crt_backend_api.c
// void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int
// dtype_code_hint,
//                                int dtype_bits_hint) {
//   tvm_crt_error_t err = kTvmErrorNoError;
//   void* ptr = 0;
//   DLDevice dev = {(DLDeviceType)device_type, device_id};
//   assert(nbytes > 0);
//   err = TVMPlatformMemoryAllocate(nbytes, dev, &ptr);
//   CHECK_EQ(err, kTvmErrorNoError,
//            "TVMBackendAllocWorkspace(%d, %d, %" PRIu64 ", %d, %d) -> %" PRId32, device_type,
//            device_id, nbytes, dtype_code_hint, dtype_bits_hint, err);
//   return ptr;
// }

// // TODO(mehrdadh): remove and reuse the CRT
// // implementation in src/runtime/crt/common/crt_backend_api.c
// int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
//   tvm_crt_error_t err = kTvmErrorNoError;
//   DLDevice dev = {(DLDeviceType)device_type, device_id};
//   err = TVMPlatformMemoryFree(ptr, dev);
//   CHECK_EQ(err, kTvmErrorNoError, "TVMBackendFreeWorkspace(%d, %d)", device_type, device_id);
//   return err;
// }

// void TVMLogf(const char* msg, ...) {
//   char buffer[128];
//   int size;
//   va_list args;
//   va_start(args, msg);
//   size = TVMPlatformFormatMessage(buffer, 128, msg, args);
//   va_end(args);
//   UartTxWrite(buffer, (size_t)size);
// }

// #ifdef __cplusplus
// }  // extern "C"
// #endif