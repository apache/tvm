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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <power/reboot.h>
#include <zephyr.h>

#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/packed_func.h>

#include "zephyr_runtime.h"

// Heap for use by TVMPlatformMemoryAllocate.
K_HEAP_DEFINE(tvm_heap, 1024 * 1024);

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                                              \
  do {                                                                               \
    tvm_crt_error_t ret = (func);                                                    \
    if (ret != kTvmErrorNoError) {                                                   \
      LOG_ERROR("%s: %d: error: %s\n", __FILE__, __LINE__, TVMGetLastError()); \
      TVMPlatformAbort(ret);                                                                     \
    }                                                                                \
  } while (0)

void* tvm_runtime_create(const char* json_data, const char* params_data,
                                 const uint64_t params_size) {
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  TVMByteArray params;
  params.data = params_data;
  params.size = params_size;

  DLDevice dev;
  dev.device_type = (DLDeviceType)device_type;
  dev.device_id = device_id;

  TVM_CCALL(TVMInitializeRuntime());
  TVMPackedFunc pf;
  TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
  TVM_CCALL(TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args));
  TVM_CCALL(TVMPackedFunc_Call(&pf));

  TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

  // run modules
  TVMGraphExecutor* graph_executor = NULL;
  TVM_CCALL(TVMGraphExecutor_Create(json_data, mod_syslib, &dev, &graph_executor));
  TVM_CCALL(TVMGraphExecutor_LoadParams(graph_executor, params.data, params.size));

  return graph_executor;
}

void tvm_runtime_destroy(void* executor) {
  TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
  TVMGraphExecutor_Release(&graph_executor);
}

void tvm_runtime_set_input(void* executor, const char* name, DLTensor* tensor) {
  TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
  TVMGraphExecutor_SetInput(graph_executor, name, tensor);
}

void tvm_runtime_run(void* executor) {
  TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
  TVMGraphExecutor_Run(graph_executor);
}

void tvm_runtime_get_output(void* executor, int32_t index, DLTensor* tensor) {
  TVMGraphExecutor* graph_executor = (TVMGraphExecutor*)executor;
  TVMGraphExecutor_GetOutput(graph_executor, index, tensor);
}

void TVMLogf(const char* msg, ...) {
  char buffer[256];
  int size;
  va_list args;
  va_start(args, msg);
  size = vsprintf(buffer, msg, args);
  va_end(args);
  write_serial(buffer, (size_t)size);
}

void TVMPlatformAbort(tvm_crt_error_t error_code) {
  if (error_code != kTvmErrorNoError) {
    uint8_t info[2];
    TVMErrorCodeTranslate(error_code, info);
    TVMLogf("TVMPlatformAbort: category: %d, code: %d\n", info[0], info[1]);
  }
  exit(-1);
}

// Called by TVM to allocate memory.
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  // This is a hack for now.
  // TODO(mehrdadh): investigate better solution.
  if (num_bytes == 0) {
    num_bytes += sizeof(int);
  }

  *out_ptr = k_heap_alloc(&tvm_heap, num_bytes, K_NO_WAIT);
  return (*out_ptr == NULL) ? kTvmErrorPlatformNoMemory : kTvmErrorNoError;
}

// Called by TVM to deallocate memory.
tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  k_heap_free(&tvm_heap, ptr);
  return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformTimerStart() { return kTvmErrorFunctionCallNotImplemented; }

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) {
  return kTvmErrorFunctionCallNotImplemented;
}
