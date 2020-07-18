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
#include <tvm/runtime/c_runtime_api.h>

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                                              \
  do {                                                                               \
    int ret = (func);                                                                \
    if (ret != 0) {                                                                  \
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

  // declare pointers
  TVMModuleHandle (*SystemLibraryCreate)();
  TVMModuleHandle (*TVMGraphRuntimeCreate)(const char*, const TVMModuleHandle, const TVMContext*);
  int (*TVMGraphRuntime_LoadParams)(TVMModuleHandle, const char*, const uint32_t);

  TVM_CCALL(TVMRuntimeInitialize());

  // get pointers
  TVM_CCALL(TVMFuncGetGlobal("runtime.SystemLib", (TVMFunctionHandle*)&SystemLibraryCreate));
  TVM_CCALL(
      TVMFuncGetGlobal("tvm.graph_runtime.create", (TVMFunctionHandle*)&TVMGraphRuntimeCreate));

  // run modules
  TVMModuleHandle mod_syslib = SystemLibraryCreate();
  TVMModuleHandle mod = TVMGraphRuntimeCreate(json_data, mod_syslib, &ctx);
  TVM_CCALL(
      TVMModGetFunction(mod, "load_params", 0, (TVMFunctionHandle*)&TVMGraphRuntime_LoadParams));
  TVMGraphRuntime_LoadParams(mod, params.data, params.size);

  return mod;
}

TVM_DLL void tvm_runtime_destroy(void* runtime) {
  void (*TVMGraphRuntimeRelease)(TVMModuleHandle*);
  TVM_CCALL(
      TVMFuncGetGlobal("tvm.graph_runtime.release", (TVMFunctionHandle*)&TVMGraphRuntimeRelease));
  TVMGraphRuntimeRelease(&runtime);
}

TVM_DLL void tvm_runtime_set_input(void* runtime, const char* name, DLTensor* tensor) {
  void (*TVMGraphRuntime_SetInput)(TVMModuleHandle, const char*, DLTensor*);
  TVM_CCALL(TVMFuncGetGlobal("tvm.graph_runtime.set_input",
                             (TVMFunctionHandle*)&TVMGraphRuntime_SetInput));
  TVMGraphRuntime_SetInput(runtime, name, tensor);
}

TVM_DLL void tvm_runtime_run(void* runtime) {
  void (*TVMGraphRuntime_Run)(TVMModuleHandle runtime);
  TVM_CCALL(TVMFuncGetGlobal("tvm.graph_runtime.run", (TVMFunctionHandle*)&TVMGraphRuntime_Run));
  TVMGraphRuntime_Run(runtime);
}

TVM_DLL void tvm_runtime_get_output(void* runtime, int32_t index, DLTensor* tensor) {
  int (*TVMGraphRuntime_GetOutput)(TVMModuleHandle, const int32_t, DLTensor*);
  TVM_CCALL(TVMFuncGetGlobal("tvm.graph_runtime.get_output",
                             (TVMFunctionHandle*)&TVMGraphRuntime_GetOutput));
  TVMGraphRuntime_GetOutput(runtime, index, tensor);
}
