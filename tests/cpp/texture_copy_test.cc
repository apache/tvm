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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/registry.h>

#include <cmath>
#include <random>

TEST(TextureCopy, HostDeviceRT) {
  using namespace tvm;
  bool enabled = tvm::runtime::RuntimeEnabled("opencl");
  if (!enabled) {
    LOG(INFO) << "Skip texture copy test because opencl runtime is disabled.\n";
    return;
  }

  std::vector<int64_t> shape{16, 16, 4};
  auto cpu_arr0 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr1 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  String mem_scope = "global.texture";
  auto opencl_txarr0 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLOpenCL, 0}, mem_scope);

  size_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    size *= static_cast<size_t>(shape[i]);
  }

  std::random_device dev;
  std::mt19937 mt(dev());
  std::uniform_real_distribution<> random(-10.0, 10.0);

  // Random initialize host ndarray
  for (size_t i = 0; i < size; i++) {
    static_cast<float*>(cpu_arr0->data)[i] = random(mt);
  }

  // Do a roundtrip from host storage to opencl texture storage and back
  cpu_arr0.CopyTo(opencl_txarr0);
  opencl_txarr0.CopyTo(cpu_arr1);
  for (size_t i = 0; i < size; ++i) {
    ICHECK_LT(
        std::fabs(static_cast<float*>(cpu_arr1->data)[i] - static_cast<float*>(cpu_arr0->data)[i]),
        1e-5);
  }
}

TEST(TextureCopy, OverwritePoolSubview) {
  using namespace tvm;
  bool enabled = tvm::runtime::RuntimeEnabled("opencl");
  if (!enabled) {
    LOG(INFO) << "Skip texture copy test because opencl runtime is disabled.\n";
    return;
  }

  std::vector<int64_t> shape{16, 16, 4};
  std::vector<int64_t> shape_pool{32, 32, 4};
  auto cpu_arr0 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_arr1 = runtime::NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_pool0 = runtime::NDArray::Empty(shape_pool, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto cpu_pool1 = runtime::NDArray::Empty(shape_pool, {kDLFloat, 32, 1}, {kDLCPU, 0});

  String mem_scope = "global.texture";
  auto opencl_txpool =
      runtime::NDArray::Empty(shape_pool, {kDLFloat, 32, 1}, {kDLOpenCL, 0}, mem_scope);
  auto opencl_txarr0 = opencl_txpool.CreateView(shape, {kDLFloat, 32, 1});

  std::random_device dev;
  std::mt19937 mt(dev());
  std::uniform_real_distribution<> random(-10.0, 10.0);

  size_t size = 1;
  size_t size_pool = 1;
  for (size_t i = 0; i < shape_pool.size(); ++i) {
    size *= static_cast<size_t>(shape[i]);
    size_pool *= static_cast<size_t>(shape_pool[i]);
  }

  // Random initialize host pool storage
  for (size_t i = 0; i < size_pool; i++) {
    static_cast<float*>(cpu_pool0->data)[i] = random(mt);
  }

  // Random initialize host array storage
  for (size_t i = 0; i < size; i++) {
    static_cast<float*>(cpu_arr0->data)[i] = random(mt);
  }

  // Loop through pool
  cpu_pool0.CopyTo(opencl_txpool);
  opencl_txpool.CopyTo(cpu_pool1);

  for (size_t i = 0; i < size_pool; i++) {
    ICHECK_LT(std::fabs(static_cast<float*>(cpu_pool0->data)[i] -
                        static_cast<float*>(cpu_pool1->data)[i]),
              1e-5);
  }

  // Loop through view
  cpu_arr0.CopyTo(opencl_txarr0);
  opencl_txarr0.CopyTo(cpu_arr1);

  for (size_t i = 0; i < size; i++) {
    ICHECK_LT(
        std::fabs(static_cast<float*>(cpu_arr0->data)[i] - static_cast<float*>(cpu_arr1->data)[i]),
        1e-5);
  }
}
