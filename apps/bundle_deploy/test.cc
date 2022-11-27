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
#ifdef _WIN32
#define NOMINMAX      // for limits
#include <Windows.h>  // LoadLibrary
#else
#include <dlfcn.h>  //dlopen
#include <sys/stat.h>
#endif
#include <tvm/runtime/c_runtime_api.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

template <typename F>
auto getFunc(void* bundle, const char* name) {
#ifdef _WIN32
  auto* f =
      reinterpret_cast<typename std::add_pointer<F>::type>(GetProcAddress((HMODULE)bundle, name));
  assert(f != nullptr);
#else
  dlerror();
  auto* f = reinterpret_cast<typename std::add_pointer<F>::type>(dlsym(bundle, name));
  assert(!dlerror());
#endif
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

template <class T>
float delta_time(T t1, T t0) {
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  return static_cast<float>(us) / 1000.f;
}

int main(int argc, char** argv) {
  assert(argc == 6 && "Usage: test <bundle.so> <data.bin> <output.bin> <graph.json> <params.bin>");
#ifdef _WIN32
  auto* bundle = LoadLibraryA(argv[1]);
#else
  auto* bundle = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
#endif
  assert(bundle);

  char* json_data;
  char* params_data;
  size_t params_size;

  json_data = read_all_or_die("json_data", argv[4], nullptr);
  params_data = read_all_or_die("params_data", argv[5], &params_size);

  auto t0 = std::chrono::steady_clock::now();

  auto* handle = getFunc<void*(char*, char*, int)>(bundle, "tvm_runtime_create")(
      json_data, params_data, params_size);
  auto t1 = std::chrono::steady_clock::now();

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
  auto t2 = std::chrono::steady_clock::now();

#ifdef _WIN32
  auto* ftvm_runtime_run =
      (auto (*)(void*)->void)GetProcAddress((HMODULE)bundle, "tvm_runtime_run");
  assert(ftvm_runtime_run != nullptr);
#else
  auto* ftvm_runtime_run = (auto (*)(void*)->void)dlsym(bundle, "tvm_runtime_run");
  assert(!dlerror());
#endif
  ftvm_runtime_run(handle);
  auto t3 = std::chrono::steady_clock::now();

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
  auto t4 = std::chrono::steady_clock::now();

  for (auto i = 0; i < 10 * 5; ++i) {
    assert(fabs(output_storage[i] - result_storage[i]) < 1e-5f);
    if (fabs(output_storage[i] - result_storage[i]) >= 1e-5f) {
      printf("got %f, expected %f\n", output_storage[i], result_storage[i]);
    }
  }

  getFunc<void(void*)>(bundle, "tvm_runtime_destroy")(handle);
  auto t5 = std::chrono::steady_clock::now();

  printf(
      "timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
      "%.2f ms (get_output), %.2f ms (destroy)\n",
      delta_time(t1, t0), delta_time(t2, t1), delta_time(t3, t2), delta_time(t4, t3),
      delta_time(t5, t4));

  free(json_data);
  free(params_data);
#ifdef _WIN32
  FreeLibrary(bundle);
#else
  dlclose(bundle);
#endif

  return 0;
}
