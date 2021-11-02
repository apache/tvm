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
 * \file main.cc
 * \brief main entry point for host subprocess-based CRT
 */
#include <inttypes.h>
#include <time.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/logging.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <unistd.h>

#include <chrono>
#include <iostream>

#include "crt_config.h"
#include "input.h"
#include "output.h"


using namespace std::chrono;

extern "C" {

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt,
                                va_list args) {
  return vsnprintf(out_buf, out_buf_size_bytes, fmt, args);
}

void __attribute__((format(printf, 1, 2))) TVMLogf(const char* format, ...){
  va_list args;
  char log_buffer[1024];

  va_start(args, format);
  size_t num_bytes_logged = TVMPlatformFormatMessage(log_buffer, sizeof(log_buffer), format, args);
  va_end(args);

  fprintf( stderr, "# %s", log_buffer);
}

void TVMPlatformAbort(tvm_crt_error_t error_code) {
  std::cerr << "TVMPlatformAbort: " << error_code << std::endl;
  throw "Aborted";
}

MemoryManagerInterface* memory_manager;

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return memory_manager->Allocate(memory_manager, num_bytes, dev, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return memory_manager->Free(memory_manager, ptr, dev);
}

steady_clock::time_point g_microtvm_start_time;
int g_microtvm_timer_running = 0;

tvm_crt_error_t TVMPlatformTimerStart() {
  if (g_microtvm_timer_running) {
    std::cerr << "timer already running" << std::endl;
    return kTvmErrorPlatformTimerBadState;
  }
  g_microtvm_start_time = std::chrono::steady_clock::now();
  g_microtvm_timer_running = 1;
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  if (!g_microtvm_timer_running) {
    std::cerr << "timer not running" << std::endl;
    return kTvmErrorPlatformTimerBadState;
  }
  auto microtvm_stop_time = std::chrono::steady_clock::now();
  std::chrono::microseconds time_span = std::chrono::duration_cast<std::chrono::microseconds>(
      microtvm_stop_time - g_microtvm_start_time);
  *elapsed_time_seconds = static_cast<double>(time_span.count()) / 1e6;
  g_microtvm_timer_running = 0;
  return kTvmErrorNoError;
}

const void* TVMSystemLibEntryPoint(void) {return NULL;}

int32_t tvmgen_default_run_model(void* arg0,void* arg1);

}

uint8_t memory[128 * 1024 * 1024];


int main(int argc, char** argv) {
  double elapsed_time_seconds;
  TVMLogf("AOT Runtime\n");

  int status =
      PageMemoryManagerCreate(&memory_manager, memory, sizeof(memory), 8   /* page_size_log2 */);
  if (status != 0) {
    fprintf(stderr, "error initiailizing memory manager\n");
    return 2;
  }

  TVMPlatformTimerStart();
  int32_t ret = tvmgen_default_run_model(input, output);
  TVMPlatformTimerStop(&elapsed_time_seconds);

  TVMLogf("elapsed_time_seconds: %f\n", elapsed_time_seconds);
  TVMLogf("Output\n");
  for (int i = 0; i < output_len; i++){
    TVMLogf("   %d, %f\n", i, output[i]);
  }

  if(ret != 0){
    TVMLogf("AOT Error");
  }

  return 0;
}
