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

#include "tvm/runtime/c_runtime_api.h"
#include <assert.h>
#include <dlfcn.h> //dlopen
#include <dlpack/dlpack.h>

int main(int argc, char **argv) {
  /* assert(argc == 2 && "Usage: demo <bundle.so>"); */
  /* auto *bundle = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL); */
  /* assert(bundle); */

  // json graph
  const char * json_fn = "build/deploy.min.json";
  const char * params_fn = "build/deploy.params";
  
  FILE * json_in = fopen(json_fn, "rt");
  /* std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>()); */
  /* json_in.close(); */
  EXPECT_NE(json_in, (void*)0);
  if (json_in == (void*)0) { exit(-1); }
  size_t json_size = findSize(json_fn);
  char * json_data = (char*)malloc(json_size+1);
  fread(json_data, json_size, 1, json_in);
  fclose(json_in);
  json_in = 0;

  // parameters in binary
  FILE * params_in = fopen(params_fn, "rb");
  EXPECT_NE(params_in, (void*)0);
  if (params_in == (void*)0) { exit(-1); }
  /* std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>()); */
  /* params_in.close(); */
  size_t params_size = findSize(params_fn);
  char * params_data = (char*)malloc(params_size);
  fread(params_data, params_size, 1, params_in);
  fclose(params_in);
  params_in = 0;

  // parameters need to be TVMByteArray type to indicate the binary data
  TVMByteArray params_arr;
  params_arr.data = params_data;
  params_arr.size = params_size;

  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;

  // get global function module for graph runtime
  TVMContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  {
    JSONReader reader;
    EXPECT_GE(sizeof(reader.is_), json_size);
  }
  GraphRuntime * runtime = TVMGraphRuntimeCreate(json_data, 0, &ctx);
  /* typedef Module * (*TVMGraphRuntimeCreateFunc)(char *, void *, int, int) ; */
  /* TVMFunctionHandle graph_runtime; */
  /* TVMGraphRuntimeCreateFunc graph_runtime; */
  /* TVMFuncGetGlobal("tvm.graph_runtime.create", (TVMFunctionHandle*)&graph_runtime); */
  /* Module * mod = graph_runtime(json_data, mod_syslib, device_type, device_id); */
  /* Module * mod = graph_runtime->pushArg(json_data)->pushArg(mod_syslib) */
  /*   ->pushArg(device_type)->pushArg(device_id)->invoke()->asModule(); */
  
  DLTensor x;
  int in_ndim = 4;
  int64_t in_shape[4] = {1, 3, 224, 224};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &x);
  
  // load image data saved in binary
  FILE * data_fin = fopen("cat.bin", "rb");
  /* data_fin.read(static_cast<char*>(x->data), 3 * 224 * 224 * 4); */
  int64_t data_size = findSize("cat.bin");
  /* char * data_data = (char*)malloc(data_size); */
  if (data_size != (in_shape[0]*in_shape[1]*in_shape[2]*in_shape[3]*4)) {
    LOGE("wrong file size: %d != %d", (int32_t)data_size, (int32_t)(in_shape[0]*in_shape[1]*in_shape[2]*in_shape[3]*4));
  }
  fread(x.data, data_size, 1, data_fin);
  fclose(data_fin);
  data_fin = 0;

  // get the function from the module(set input data)
  // TVMSetInputCFunc set_input = mod->GetFunction("set_input");
  // mod->set_input(mod, "data", &x);
  runtime->SetInput(runtime, "data", &x);

  // get the function from the module(load patameters)
  // TVMLoadParamsCFunc load_params = mod->GetFunction("load_params");
  runtime->LoadParams(runtime, params_arr.data, params_arr.size);

  // get the function from the module(run it)
  // TVMRunCFunc run = mod->GetFunction("run");
  runtime->Run(runtime);

//  for (int idx = 0; idx < 1000; idx++) {
//    LOGI("%d: %.3f", idx, ((float*)runtime->op_execs[263].args.values[1].v_handle->data)[idx]);
//  }
  
  DLTensor y;
  int out_ndim = 1;
  int64_t out_shape[1] = {1000, };
  TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

  // get the function from the module(get output data)
  // TVMPackedCFunc get_output = mod->GetFunction("get_output");
  /* get_output(0, y); */
  runtime->GetOutput(runtime, 0, &y);

  // get the maximum position in output vector
  // auto y_iter = static_cast<float*>(y->data);
  // auto max_iter = std::max_element(y_iter, y_iter + 1000);
  // auto max_index = std::distance(y_iter, max_iter);
  float max_iter = - FLT_MAX;
  int32_t max_index = -1;
  float * y_iter = (float*)y.data;
  // qsort(y_iter, 1000, sizeof(float), compareFloat);
  for (int32_t idx = 0; idx < 1000; idx++) {
    if (y_iter[idx] > max_iter) {
      max_iter = y_iter[idx];
      max_index = idx;
    }
  }
  
  LOGI("The maximum position in output vector is: %d, with max-value %f.", max_index, max_iter);

  TVMArrayFree(&y);
  TVMArrayFree(&x);

  return 0;
}
