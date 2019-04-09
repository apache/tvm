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
#include <iostream>
#include <random>
#include <vector>

template <typename F> auto getFunc(void *bundle, const char *name) {
  dlerror();
  auto *f =
      reinterpret_cast<typename std::add_pointer<F>::type>(dlsym(bundle, name));
  assert(!dlerror());
  return f;
}

int main(int argc, char **argv) {
  assert(argc == 2 && "Usage: demo <bundle.so>");
  auto *bundle = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
  assert(bundle);

  auto *handle = getFunc<void *()>(bundle, "tvm_runtime_create")();

  std::vector<float> input_storage(1 * 3 * 224 * 224);
  std::mt19937 gen(0);
  for (auto &e : input_storage) {
    e = std::uniform_real_distribution<float>(0.0, 1.0)(gen);
  }

  std::vector<int64_t> input_shape = {1, 3, 224, 224};
  DLTensor input;
  input.data = input_storage.data();
  input.ctx = DLContext{kDLCPU, 0};
  input.ndim = 4;
  input.dtype = DLDataType{kDLFloat, 32, 1};
  input.shape = input_shape.data();
  input.strides = nullptr;
  input.byte_offset = 0;
  getFunc<void(void *, const char *, void *)>(bundle, "tvm_runtime_set_input")(
      handle, "data", &input);

  auto *ftvm_runtime_run =
      (auto (*)(void *)->void)dlsym(bundle, "tvm_runtime_run");
  assert(!dlerror());
  ftvm_runtime_run(handle);

  std::vector<float> output_storage(1000);
  std::vector<int64_t> output_shape = {1, 1000};
  DLTensor output;
  output.data = output_storage.data();
  output.ctx = DLContext{kDLCPU, 0};
  output.ndim = 2;
  output.dtype = DLDataType{kDLFloat, 32, 1};
  output.shape = output_shape.data();
  output.strides = nullptr;
  output.byte_offset = 0;

  getFunc<void(void *, int, void *)>(bundle, "tvm_runtime_get_output")(
      handle, 0, &output);
  for (auto i = 0; i < output_storage.size(); ++i) {
    std::cerr << "output[" << i << "]: " << output_storage[i] << std::endl;
  }
  getFunc<void(void *)>(bundle, "tvm_runtime_destroy")(handle);
  dlclose(bundle);
  return 0;
}
