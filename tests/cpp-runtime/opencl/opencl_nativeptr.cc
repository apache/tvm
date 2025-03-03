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

#include <gtest/gtest.h>
#include <tvm/runtime/container/optional.h>

#include <cmath>
#include <random>

#include "../src/runtime/opencl/opencl_common.h"

using namespace tvm::runtime;
using namespace tvm::runtime::cl;

#if defined(OPENCL_ENABLE_HOST_PTR)
TEST(OpenCLNativePtr, access_memory) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();

  auto A = tvm::runtime::NDArray::Empty({128, 128}, {kDLFloat, 32, 1}, {kDLOpenCL, 0});
  void* nptr = workspace->GetNativePtr(A);
  memset(nptr, 0x0, 128 * 128 * 4);
}

TEST(OpenCLNatvePtr, data_loop) {
  OpenCLWorkspace* workspace = OpenCLWorkspace::Global();

  auto cl_arr = tvm::runtime::NDArray::Empty({1024}, {kDLFloat, 32, 1}, {kDLOpenCL, 0});
  auto cpu_arr = tvm::runtime::NDArray::Empty({1024}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  std::random_device rdev;
  std::mt19937 mt(rdev());
  std::uniform_real_distribution<> random(-10.0, 10.0);

  // Random initialize host ndarray
  for (size_t i = 0; i < 1024; i++) {
    static_cast<float*>(cpu_arr->data)[i] = random(mt);
  }
  // Do a roundtrip from cpu arr to opencl array and native ptr.
  cpu_arr.CopyTo(cl_arr);
  void* nptr = workspace->GetNativePtr(cl_arr);
  for (size_t i = 0; i < 1024; ++i) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] - static_cast<float*>(nptr)[i]),
              1e-5);
  }

  // Random initialize cl ndarray
  for (size_t i = 0; i < 1024; i++) {
    static_cast<float*>(nptr)[i] = random(mt);
  }
  // Do a roundtrip from native ptr to cl arr to cpu array.
  cl_arr.CopyTo(cpu_arr);
  for (size_t i = 0; i < 1024; ++i) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_arr->data)[i] - static_cast<float*>(nptr)[i]),
              1e-5);
  }
}

#endif
