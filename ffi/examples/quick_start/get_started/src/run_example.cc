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
#include <tvm/ffi/container/ndarray.h>
#include <tvm/ffi/extra/module.h>

// This file shows how to load the same compiled module and interact with it in C++
namespace ffi = tvm::ffi;

struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) { tensor->data = malloc(ffi::GetDataSize(*tensor)); }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

inline ffi::NDArray Empty(ffi::Shape shape, DLDataType dtype, DLDevice device) {
  return ffi::NDArray::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
}

int main() {
  // load the module
  ffi::Module mod = ffi::Module::LoadFromFile("build/add_one_cpu.so");

  // create an NDArray, alternatively, one can directly pass in a DLTensor*
  ffi::NDArray x = Empty({5}, DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  for (int i = 0; i < 5; ++i) {
    reinterpret_cast<float*>(x->data)[i] = static_cast<float>(i);
  }

  ffi::Function add_one_cpu = mod->GetFunction("add_one_cpu").value();
  add_one_cpu(x, x);

  std::cout << "x after add_one_cpu(x, x)" << std::endl;
  for (int i = 0; i < 5; ++i) {
    std::cout << reinterpret_cast<float*>(x->data)[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}
