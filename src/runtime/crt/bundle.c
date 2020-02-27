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

#include "graph_runtime.h"

TVM_DLL TVMGraphRuntime * tvm_runtime_create(const char * json_data,
                                             const char * params_data,
                                             const uint64_t params_size) {
  int64_t device_type = kDLCPU;
  int64_t device_id = 0;

  TVMByteArray params;
  params.data = params_data;
  params.size = params_size;

  TVMContext ctx;
  ctx.device_type = device_type;
  ctx.device_id = device_id;
  TVMGraphRuntime * runtime = TVMGraphRuntimeCreate(json_data, 0, &ctx);

  runtime->LoadParams(runtime, params.data, params.size);

  return runtime;
}

TVM_DLL void tvm_runtime_destroy(TVMGraphRuntime * runtime) {
  TVMGraphRuntimeRelease(&runtime);
}

TVM_DLL void tvm_runtime_set_input(TVMGraphRuntime * runtime, const char * name,
                                   DLTensor * tensor) {
  runtime->SetInput(runtime, "data", tensor);
}

TVM_DLL void tvm_runtime_run(TVMGraphRuntime * runtime) {
  runtime->Run(runtime);
}

TVM_DLL void tvm_runtime_get_output(TVMGraphRuntime * runtime, int32_t index,
                                    DLTensor * tensor) {
  runtime->GetOutput(runtime, index, tensor);
}

