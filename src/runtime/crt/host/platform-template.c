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
#include <tvm/runtime/crt/error_codes.h>
#include <tvm/runtime/crt/page_allocator.h>

uint8_t memory[MEMORY_SIZE_BYTES];
MemoryManagerInterface* memory_manager;

// Called by TVM when an internal invariant is violated, and execution cannot continue.
__attribute__((weak)) void TVMPlatformAbort(tvm_crt_error_t error_code) { exit(1); }

// Called by TVM when a message needs to be formatted.
__attribute__((weak)) size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes,
                                                      const char* fmt, va_list args) {
  return vsprintf(out_buf, fmt, args);
}

// Called by TVM when memory allocation is required.
__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev,
                                                                void** out_ptr) {
  return memory_manager->Allocate(memory_manager, num_bytes, dev, out_ptr);
}

// Called by TVM to free an allocated memory.
__attribute__((weak)) tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return memory_manager->Free(memory_manager, ptr, dev);
}

__attribute__((weak)) tvm_crt_error_t TVMPlatformTimerStart() { return kTvmErrorNoError; }

__attribute__((weak)) tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  return kTvmErrorNoError;
}
__attribute__((weak)) tvm_crt_error_t TVMPlatformBeforeMeasurement() { return kTvmErrorNoError; }

__attribute__((weak)) tvm_crt_error_t TVMPlatformAfterMeasurement() { return kTvmErrorNoError; }

__attribute__((weak)) tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  return kTvmErrorNoError;
}

__attribute__((weak)) tvm_crt_error_t TVMPlatformInitialize() {
  int status =
      PageMemoryManagerCreate(&memory_manager, memory, sizeof(memory), 8 /* page_size_log2 */);
  if (status != 0) {
    fprintf(stderr, "error initiailizing memory manager\n");
    return kTvmErrorPlatformMemoryManagerInitialized;
  }
  return kTvmErrorNoError;
}
