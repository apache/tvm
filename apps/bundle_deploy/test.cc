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
#include <sys/stat.h>
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

char* read_all_or_die(const char* name, const char* file_path, size_t* out_size) {
  struct stat st;
  if (stat(file_path, &st)) {
    char err[1024];
    snprintf(err, sizeof(err), "%s: statting file", name);
    perror(err);
    abort();
  }
  if (st.st_size > 1024 * 1024) {
    std::cerr << name << ": file is over 1MB limit: " << st.st_size << " bytes" << std::endl;
    abort();
  }

  if (out_size != nullptr) {
    *out_size = st.st_size;
  }

  char* data = (char*)malloc(st.st_size);
  FILE* fp = fopen(file_path, "rb");
  size_t bytes_to_read = st.st_size;
  size_t bytes_read = 0;
  while (bytes_read < bytes_to_read) {
    size_t this_round = fread(data, 1, st.st_size, fp);
    if (this_round == 0) {
      if (ferror(fp)) {
        char err[1024];
        snprintf(err, sizeof(err), "%s: error during read", name);
        perror(err);
      } else if (feof(fp)) {
        std::cerr << name << ": file is shorter than its stat size (" << bytes_read << " v "
                  << st.st_size << ")" << std::endl;
      } else {
        std::cerr << name << ": fread stopped returning data" << std::endl;
      }
      abort();
    }
    bytes_read += this_round;
  }

  fclose(fp);
  return data;
}

int main(int argc, char** argv) {
  assert(argc == 6 && "Usage: test <bundle.so> <data.bin> <output.bin> <graph.json> <params.bin>");
  auto* bundle = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
  assert(bundle);

  char* json_data;
  char* params_data;
  size_t params_size;

  json_data = read_all_or_die("json_data", argv[4], nullptr);
  params_data = read_all_or_die("params_data", argv[5], &params_size);

  struct timeval t0, t1, t2, t3, t4, t5;
  gettimeofday(&t0, 0);

  auto* handle = getFunc<void*(char*, char*, int)>(bundle, "tvm_runtime_create")(
      json_data, params_data, params_size);
  gettimeofday(&t1, 0);

  size_t input_storage_size;
  float* input_storage =
      reinterpret_cast<float*>(read_all_or_die("input_storage", argv[2], &input_storage_size));
  size_t result_storage_size;
  float* result_storage =
      reinterpret_cast<float*>(read_all_or_die("result_storage", argv[3], &result_storage_size));

  size_t expected_size = 10 * 5 * sizeof(float);
  if (input_storage_size != expected_size || result_storage_size != expected_size) {
    std::cerr << "wrong input or result storage size (want " << expected_size
              << "input_storage_size=" << input_storage_size
              << "; result_storage_size=" << result_storage_size << std::endl;
  }

  std::vector<int64_t> input_shape = {10, 5};
  DLTensor input;
  input.data = input_storage;
  input.device = DLDevice{kDLCPU, 0};
  input.ndim = 2;
  input.dtype = DLDataType{kDLFloat, 32, 1};
  input.shape = input_shape.data();
  input.strides = nullptr;
  input.byte_offset = 0;

  getFunc<void(void*, const char*, void*)>(bundle, "tvm_runtime_set_input")(handle, "x", &input);
  gettimeofday(&t2, 0);

  auto* ftvm_runtime_run = (auto(*)(void*)->void)dlsym(bundle, "tvm_runtime_run");
  assert(!dlerror());
  ftvm_runtime_run(handle);
  gettimeofday(&t3, 0);

  float output_storage[10 * 5];
  std::vector<int64_t> output_shape = {10, 5};
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

  for (auto i = 0; i < 10 * 5; ++i) {
    assert(fabs(output_storage[i] - result_storage[i]) < 1e-5f);
    if (fabs(output_storage[i] - result_storage[i]) >= 1e-5f) {
      printf("got %f, expected %f\n", output_storage[i], result_storage[i]);
    }
  }

  getFunc<void(void*)>(bundle, "tvm_runtime_destroy")(handle);
  gettimeofday(&t5, 0);

  printf(
      "timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
      "%.2f ms (get_output), %.2f ms (destroy)\n",
      (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.f,
      (t2.tv_sec - t1.tv_sec) * 1000.0f + (t2.tv_usec - t1.tv_usec) / 1000.f,
      (t3.tv_sec - t2.tv_sec) * 1000.0f + (t3.tv_usec - t2.tv_usec) / 1000.f,
      (t4.tv_sec - t3.tv_sec) * 1000.0f + (t4.tv_usec - t3.tv_usec) / 1000.f,
      (t5.tv_sec - t4.tv_sec) * 1000.0f + (t5.tv_usec - t4.tv_usec) / 1000.f);

  free(json_data);
  free(params_data);
  dlclose(bundle);

  return 0;
}
