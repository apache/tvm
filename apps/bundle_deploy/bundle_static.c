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

#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/crt/crt.h>
#include <tvm/runtime/crt/graph_runtime.h>
#include <tvm/runtime/crt/packed_func.h>

#include "bundle.h"

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                                              \
  do {                                                                               \
    tvm_crt_error_t ret = (func);                                                    \
    if (ret != kTvmErrorNoError) {                                                   \
      fprintf(stderr, "%s: %d: error: %s\n", __FILE__, __LINE__, TVMGetLastError()); \
      exit(ret);                                                                     \
    }                                                                                \
  } while (0)

TVM_DLL void* tvm_runtime_create(const char* json_data, const char* params_data,
                                 const uint64_t params_size) {
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  TVMByteArray params;
  params.data = params_data;
  params.size = params_size;

  TVMContext ctx;
  ctx.device_type = (DLDeviceType)device_type;
  ctx.device_id = device_id;

  // get pointers
  TVM_CCALL(TVMInitializeRuntime());
  TVMPackedFunc pf;
  TVMArgs args = TVMArgs_Create(NULL, NULL, 0);
  TVM_CCALL(TVMPackedFunc_InitGlobalFunc(&pf, "runtime.SystemLib", &args));
  TVM_CCALL(TVMPackedFunc_Call(&pf));

  TVMModuleHandle mod_syslib = TVMArgs_AsModuleHandle(&pf.ret_value, 0);

  // run modules
  TVMGraphRuntime* graph_runtime = TVMGraphRuntime_Create(json_data, mod_syslib, &ctx);
  TVMGraphRuntime_LoadParams(graph_runtime, params.data, params.size);

  return graph_runtime;
}

TVM_DLL void tvm_runtime_destroy(void* runtime) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_Release(&graph_runtime);
}

TVM_DLL void tvm_runtime_set_input(void* runtime, const char* name, DLTensor* tensor) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_SetInput(graph_runtime, name, tensor);
}

TVM_DLL void tvm_runtime_run(void* runtime) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_Run(graph_runtime);
}

TVM_DLL void tvm_runtime_get_output(void* runtime, int32_t index, DLTensor* tensor) {
  TVMGraphRuntime* graph_runtime = (TVMGraphRuntime*)runtime;
  TVMGraphRuntime_GetOutput(graph_runtime, index, tensor);
}

void __attribute__((noreturn)) TVMPlatformAbort(int error_code) {
  fprintf(stderr, "TVMPlatformAbort: %d\n", error_code);
  exit(-1);
}
