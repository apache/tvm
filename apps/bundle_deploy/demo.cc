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
#include <dlfcn.h>  //dlopen
#include <sys/time.h>
#include <tvm/runtime/c_runtime_api.h>

#include <iostream>
#include <random>
#include <vector>

template <typename F>
auto getFunc(void* bundle, const char* name) {
  dlerror();
  auto* f = reinterpret_cast<typename std::add_pointer<F>::type>(dlsym(bundle, name));
  assert(!dlerror());
  return f;
}

static int read_all(const char* file_description, const char* file_path, char** out_params,
                    size_t* params_size) {
  FILE* fp = fopen(file_path, "rb");
  if (fp == NULL) {
    return 2;
  }

  int error = 0;
  error = fseek(fp, 0, SEEK_END);
  if (error < 0) {
    return error;
  }

  long file_size = ftell(fp);
  if (file_size < 0) {
    return (int)file_size;
  } else if (file_size == 0 || file_size > (10 << 20)) {  // file size should be in (0, 20MB].
    char buf[128];
    snprintf(buf, sizeof(buf), "determing file size: %s", file_path);
    perror(buf);
    return 2;
  }

  if (params_size != NULL) {
    *params_size = file_size;
  }

  error = fseek(fp, 0, SEEK_SET);
  if (error < 0) {
    return error;
  }

  *out_params = (char*)malloc((unsigned long)file_size);
  if (fread(*out_params, file_size, 1, fp) != 1) {
    free(*out_params);
    *out_params = NULL;

    char buf[128];
    snprintf(buf, sizeof(buf), "reading: %s", file_path);
    perror(buf);
    return 2;
  }

  error = fclose(fp);
  if (error != 0) {
    free(*out_params);
    *out_params = NULL;
  }

  return 0;
}

int main(int argc, char** argv) {
  assert(argc == 5 && "Usage: demo <bundle.so> <graph.json> <params.bin> <cat.bin>");
  auto* bundle = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
  assert(bundle);

  char* json_data;
  int error = read_all("graph.json", argv[2], &json_data, NULL);
  if (error != 0) {
    return error;
  }

  char* params_data;
  size_t params_size;
  error = read_all("params.bin", argv[3], &params_data, &params_size);
  if (error != 0) {
    return error;
  }

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  auto* handle = getFunc<void*(char*, char*, int)>(bundle, "tvm_runtime_create")(
      json_data, params_data, params_size);
  gettimeofday(&t1, 0);

  float input_storage[1 * 3 * 224 * 224];
  FILE* fp = fopen(argv[4], "rb");
  fread(input_storage, 3 * 224 * 224, 4, fp);
  fclose(fp);

  std::vector<int64_t> input_shape = {1, 3, 224, 224};
  DLTensor input;
  input.data = input_storage;
  input.device = DLDevice{kDLCPU, 0};
  input.ndim = 4;
  input.dtype = DLDataType{kDLFloat, 32, 1};
  input.shape = input_shape.data();
  input.strides = nullptr;
  input.byte_offset = 0;

  getFunc<void(void*, const char*, void*)>(bundle, "tvm_runtime_set_input")(handle, "data", &input);
  gettimeofday(&t2, 0);

  auto* ftvm_runtime_run = (auto(*)(void*)->void)dlsym(bundle, "tvm_runtime_run");
  assert(!dlerror());
  ftvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  float output_storage[1000];
  std::vector<int64_t> output_shape = {1, 1000};
  DLTensor output;
  output.data = output_storage;
  output.device = DLDevice{kDLCPU, 0};
  output.ndim = 2;
  output.dtype = DLDataType{kDLFloat, 32, 1};
  output.shape = output_shape.data();
  output.strides = nullptr;
  output.byte_offset = 0;

  getFunc<void(void*, int, void*)>(bundle, "tvm_runtime_get_output")(handle, 0, &output);
  gettimeofday(&t4, 0);

  float max_iter = -std::numeric_limits<float>::max();
  int32_t max_index = -1;
  for (auto i = 0; i < 1000; ++i) {
    if (output_storage[i] > max_iter) {
      max_iter = output_storage[i];
      max_index = i;
    }
  }

  getFunc<void(void*)>(bundle, "tvm_runtime_destroy")(handle);
  gettimeofday(&t5, 0);

  printf("The maximum position in output vector is: %d, with max-value %f.\n", max_index, max_iter);
  printf(
      "timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
      "%.2f ms (get_output), %.2f ms (destroy)\n",
      (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.f,
      (t2.tv_sec - t1.tv_sec) * 1000.0f + (t2.tv_usec - t1.tv_usec) / 1000.f,
      (t3.tv_sec - t2.tv_sec) * 1000.0f + (t3.tv_usec - t2.tv_usec) / 1000.f,
      (t4.tv_sec - t3.tv_sec) * 1000.0f + (t4.tv_usec - t3.tv_usec) / 1000.f,
      (t5.tv_sec - t4.tv_sec) * 1000.0f + (t5.tv_usec - t4.tv_usec) / 1000.f);
  dlclose(bundle);

  return 0;
}
