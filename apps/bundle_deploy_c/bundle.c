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

/* #include <memory> */
/* #include <tvm/runtime/c_runtime_api.h> */
/* #include <tvm/runtime/registry.h> */
#include <tvm/runtime/crt/graph_runtime.h>

extern unsigned char build_graph_json[];
extern unsigned int build_graph_json_len;
extern unsigned char build_params_bin[];
extern unsigned int build_params_bin_len;

#define TVM_BUNDLE_FUNCTION __attribute__((visibility("default")))

TVM_BUNDLE_FUNCTION GraphRuntime * tvm_runtime_create() {
  char * json_data = build_graph_json;

  int device_type = kDLCPU;
  int device_id = 0;

  TVMByteArray params;
  params.data = build_params_bin;
  params.size = build_params_bin_len;

  TVMContext ctx;
  ctx.device_type = device_type;
  ctx.device_id = device_id;
  GraphRuntime * runtime = TVMGraphRuntimeCreate(json_data, 0, &ctx);

  runtime->LoadParams(runtime, params.data, params.size);

  return runtime;
}

TVM_BUNDLE_FUNCTION void tvm_runtime_destroy(GraphRuntime * runtime) {
  TVMGraphRuntimeRelease(&runtime);
}

TVM_BUNDLE_FUNCTION void tvm_runtime_set_input(GraphRuntime * runtime, const char * name,
                                               DLTensor * tensor) {
  runtime->SetInput(runtime, "data", tensor);
}

TVM_BUNDLE_FUNCTION void tvm_runtime_run(GraphRuntime * runtime) {
  runtime->Run(runtime);
}

TVM_BUNDLE_FUNCTION void tvm_runtime_get_output(GraphRuntime * runtime, int32_t index,
                                                DLTensor * tensor) {
  runtime->GetOutput(runtime, index, tensor);
}

