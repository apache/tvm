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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>

#include "graph_runtime.h"
#include "ndarray.h"
#include "packed_func.h"

// Handle internal errors

static char g_last_error[1024];

void TVMAPISetLastError(const char* msg) {
  assert(strlen(msg) < sizeof(g_last_error));
  snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
}

const char* TVMGetLastError(void) { return g_last_error; }

// Manipulate NDArray on target device

int TVMArrayAlloc(const tvm_index_t* shape, int ndim, int dtype_code, int dtype_bits,
                  int dtype_lanes, int device_type, int device_id, TVMArrayHandle* out) {
  DLDataType dtype;
  dtype.code = dtype_code;
  dtype.bits = dtype_bits;
  dtype.lanes = dtype_lanes;
  DLContext ctx;
  ctx.device_type = (DLDeviceType)device_type;
  ctx.device_id = device_id;
  TVMNDArray arr = TVMNDArray_Empty(ndim, shape, dtype, ctx);
  **out = arr.dl_tensor;
  return 0;
}

int TVMArrayFree(TVMArrayHandle handle) {
  TVMNDArray arr;
  arr.dl_tensor = *handle;
  return TVMNDArray_Release(&arr);
}

void* SystemLibraryCreate() { return 0; }

int TVMModGetFunction(TVMModuleHandle mod, const char* func_name, int query_imports,
                      TVMFunctionHandle* out) {
  int status = 0;
  if (!strcmp(func_name, "load_params")) {
    *out = &TVMGraphRuntime_LoadParams;
  } else {
    status = -1;
  }
  return status;
}

int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {
  int status = 0;
  if (!strcmp(name, "tvm.graph_runtime.create")) {
    *out = &TVMGraphRuntimeCreate;
  } else if (!strcmp(name, "tvm.graph_runtime.set_input")) {
    *out = &TVMGraphRuntime_SetInput;
  } else if (!strcmp(name, "tvm.graph_runtime.run")) {
    *out = &TVMGraphRuntime_Run;
  } else if (!strcmp(name, "tvm.graph_runtime.get_output")) {
    *out = &TVMGraphRuntime_GetOutput;
  } else if (!strcmp(name, "tvm.graph_runtime.release")) {
    *out = &TVMGraphRuntimeRelease;
  } else if (!strcmp(name, "runtime.SystemLib")) {
    *out = &SystemLibraryCreate;
  } else {
    char msg[200];
    snprintf(msg, sizeof(msg), "fail to get global: name=%s", name);
    TVMAPISetLastError(msg);
    status = -1;
  }
  return status;
}
