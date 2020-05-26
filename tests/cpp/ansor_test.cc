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

#include <tvm/te/operation.h>
#include <topi/nn.h>
#include "../../src/ansor/compute_dag.h"

tvm::Array<tvm::te::Tensor> matmul_func(int n, int m, int k) {
  using namespace tvm;
  using namespace tvm::te;

  Tensor A = placeholder({n, k}, DataType::Float(32), "A");
  Tensor B = placeholder({k, m}, DataType::Float(32), "B");
  IterVar K = IterVarNode::make({0, k}, Var("k"), kCommReduce);
  const auto& C = compute(
      {n, m},
      [&](Var i, Var j) { return tvm::sum(A[i][K] * B[K][j], {K}); },
      "C");

  return {A, B, C};
}

tvm::Array<tvm::te::Tensor> conv2d_nchw_bn_relu_func(int N, int H, int W,
    int CI, int CO, int kernel_size, int strides, int padding,
    int dilation = 1) {
  using namespace tvm;
  using namespace tvm::te;

  Tensor data = placeholder({N, CI, H, W}, DataType::Float(32), "Data");
  Tensor kernel = placeholder({CO, CI, kernel_size, kernel_size},
                              DataType::Float(32), "Kernel");
  Tensor bias = placeholder({CO, 1, 1}, DataType::Float(32), "Bias");
  Tensor bn_scale = placeholder({CO, 1, 1}, DataType::Float(32), "Bn_scale");
  Tensor bn_offset = placeholder({CO, 1, 1}, DataType::Float(32), "Bn_offset");

  int OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1);
  int OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1);

  const auto& conv = topi::conv2d_nchw(data, kernel, strides, padding,
                                       dilation);
  const auto& bias_add = compute(
      {N, CO, OH, OW},
      [&](Var i, Var j, Var k, Var l) {
          return conv[i][j][k][l] + bias[j][0][0];
      },
      "Bias_add");
  const auto& bn_mul = compute(
      {N, CO, OH, OW},
      [&](Var i, Var j, Var k, Var l) {
          return bias_add[i][j][k][l] * bn_scale[j][0][0];
      },
      "Bn_mul");
  const auto& bn_add = compute(
      {N, CO, OH, OW},
      [&](Var i, Var j, Var k, Var l) {
          return bn_mul[i][j][k][l] + bn_offset[j][0][0];
      },
      "Bn_add");
  const auto& out = topi::relu<float>(bn_add);

  return {data, kernel, bias, bn_scale, bn_offset, out};
}

TEST(ComputeDAG, Basic) {
  const auto& tensors = conv2d_nchw_bn_relu_func(1, 224, 224, 3, 64, 7, 2, 3);
  auto dag = tvm::ansor::ComputeDAGNode::make(tensors);

  LOG(INFO) << "\n" << dag;
  LOG(INFO) << "\n" << dag->access_analyzer;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
