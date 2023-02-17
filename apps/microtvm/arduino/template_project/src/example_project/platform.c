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

#include "Arduino.h"
#include "standalone_crt/include/dlpack/dlpack.h"
#include "standalone_crt/include/tvm/runtime/crt/stack_allocator.h"

#define TVM_WORKSPACE_SIZE_BYTES $workspace_size_bytes

// AOT memory array, stack allocator wants it aligned
static uint8_t g_aot_memory[TVM_WORKSPACE_SIZE_BYTES]
    __attribute__((aligned(TVM_RUNTIME_ALLOC_ALIGNMENT_BYTES)));
tvm_workspace_t app_workspace;

// Called when an internal error occurs and execution cannot continue.
// Blink code for debugging purposes
void TVMPlatformAbort(tvm_crt_error_t error) {
  TVMLogf("TVMPlatformAbort: 0x%08x\n", error);
  for (;;) {
#ifdef LED_BUILTIN
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(750);
#endif
  }
}

// Allocate memory for use by TVM.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return StackMemoryManager_Allocate(&app_workspace, num_bytes, out_ptr);
}

// Free memory used by TVM.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return StackMemoryManager_Free(&app_workspace, ptr);
}

// Internal logging API call implementation.
void TVMLogf(const char* msg, ...) {}

unsigned long g_utvm_start_time_micros;
int g_utvm_timer_running = 0;

// Start a device timer.
tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 1;
  g_utvm_start_time_micros = micros();
  return kTvmErrorNoError;
}

// Stop the running device timer and get the elapsed time (in microseconds).
tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_utvm_timer_running) {
    return kTvmErrorPlatformTimerBadState;
  }
  g_utvm_timer_running = 0;
  unsigned long g_utvm_stop_time = micros() - g_utvm_start_time_micros;
  *elapsed_time_seconds = ((double)g_utvm_stop_time) / 1e6;
  return kTvmErrorNoError;
}

// Fill a buffer with random data.
tvm_crt_error_t TVMPlatformGenerateRandom(uint8_t* buffer, size_t num_bytes) {
  for (size_t i = 0; i < num_bytes; i++) {
    buffer[i] = rand();
  }
  return kTvmErrorNoError;
}

// Initialize TVM inference.
tvm_crt_error_t TVMPlatformInitialize() {
  StackMemoryManager_Init(&app_workspace, g_aot_memory, sizeof(g_aot_memory));
  return kTvmErrorNoError;
}

void TVMExecute(void* input_data, void* output_data) {
  int ret_val = tvmgen_default___tvm_main__(input_data, output_data);
  if (ret_val != 0) {
    TVMPlatformAbort(kTvmErrorPlatformCheckFailure);
  }
}
