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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <stdarg.h>
#include <tvm/runtime/crt/platform.h>

// Provide dummy implementations for TVM runtime functions for use by the tests.

extern "C" {

void InternalTVMPlatformAbort(tvm_crt_error_t error_code) {
  FAIL() << "TVMPlatformAbort(" << error_code << ")";
}

void TVMPlatformAbort(tvm_crt_error_t error_code) {
  InternalTVMPlatformAbort(error_code);
  exit(2);  // for __attribute__((noreturn))
}

struct TVMModule;
const TVMModule* TVMSystemLibEntryPoint(void) { return NULL; }

void TVMLogf(const char* fmt, ...) {
  va_list args;
  char log_buf[1024];
  va_start(args, fmt);
  int ret = vsnprintf(log_buf, sizeof(log_buf), fmt, args);
  va_end(args);

  if (ret < 0) {
    LOG(ERROR) << "TVMLogf: error formatting: " << fmt;
  } else {
    LOG(INFO) << "TVMLogf: " << std::string(log_buf, ret);
  }
}

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  *out_ptr = malloc(num_bytes);
  return *out_ptr ? kTvmErrorNoError : kTvmErrorPlatformNoMemory;
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  if (ptr) {
    free(ptr);
  }
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStart() { return kTvmErrorFunctionCallNotImplemented; }

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  return kTvmErrorFunctionCallNotImplemented;
}
}
