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
#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/runtime/container/array.h>
#include <tvm/te/operation.h>
#include <tvm/topi/nn.h>

#include <unordered_set>

// Compute declaration for test
tvm::Array<tvm::te::Tensor> conv2d_nchw_bn_relu_func(int N, int H, int W, int CI, int CO,
                                                     int kernel_size, int strides, int padding,
                                                     int dilation = 1) {
  using namespace tvm;
  using namespace tvm::te;

  Tensor data = placeholder({N, CI, H, W}, DataType::Float(32), "Data");
  Tensor kernel = placeholder({CO, CI, kernel_size, kernel_size}, DataType::Float(32), "Kernel");
  Tensor bias = placeholder({CO, 1, 1}, DataType::Float(32), "Bias");
  Tensor bn_scale = placeholder({CO, 1, 1}, DataType::Float(32), "Bn_scale");
  Tensor bn_offset = placeholder({CO, 1, 1}, DataType::Float(32), "Bn_offset");

  int OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) / strides + 1;
  int OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) / strides + 1;

  const auto& conv = topi::conv2d_nchw(data, kernel, padding, padding, strides, strides);
  ICHECK(conv->shape[2].as<IntImmNode>()->value == OH);
  ICHECK(conv->shape[3].as<IntImmNode>()->value == OW);

  const auto& bias_add = compute(
      {N, CO, OH, OW}, [&](Var i, Var j, Var k, Var l) { return conv[i][j][k][l] + bias[j][0][0]; },
      "Bias_add");
  const auto& bn_mul = compute(
      {N, CO, OH, OW},
      [&](Var i, Var j, Var k, Var l) { return bias_add[i][j][k][l] * bn_scale[j][0][0]; },
      "Bn_mul");
  const auto& bn_add = compute(
      {N, CO, OH, OW},
      [&](Var i, Var j, Var k, Var l) { return bn_mul[i][j][k][l] + bn_offset[j][0][0]; },
      "Bn_add");
  const auto& out = topi::relu<float>(bn_add);

  return {data, kernel, bias, bn_scale, bn_offset, out};
}

using namespace tvm::auto_scheduler;

// Test Access Analyzer
TEST(ComputeDAG, AccessAnalyzer) {
  const auto& tensors = conv2d_nchw_bn_relu_func(1, 224, 224, 3, 64, 7, 2, 3);
  const auto& dag = tvm::auto_scheduler::ComputeDAG(tensors);
  State s0 = dag->init_state;

  int data = 0, padding = 1, kernel = 2, conv = 3, bias = 4, bias_add = 5;
  int bn_scale = 6, bn_mul = 7, bn_offset = 8, bn_add = 9, relu = 10;

  std::set<int> needs_multi_level_tiling = {conv};
  for (size_t stage_id = 0; stage_id < dag->ops.size(); stage_id++) {
    if (needs_multi_level_tiling.count(stage_id)) {
      ICHECK(dag->access_analyzer.NeedsMultiLevelTiling(dag->ops[stage_id]));
    } else {
      ICHECK(!dag->access_analyzer.NeedsMultiLevelTiling(dag->ops[stage_id]));
    }
  }

  std::set<int> is_simple_access = {data,     padding, kernel,    bias,   bias_add,
                                    bn_scale, bn_mul,  bn_offset, bn_add, relu};
  for (size_t stage_id = 0; stage_id < dag->ops.size(); stage_id++) {
    if (is_simple_access.count(stage_id)) {
      ICHECK(dag->access_analyzer.IsSimpleAccess(dag->ops[stage_id]));
    } else {
      ICHECK(!dag->access_analyzer.IsSimpleAccess(dag->ops[stage_id]));
    }
  }

  std::set<int> is_strictly_inlinable = {bias_add, bn_mul, bn_add, relu};
  for (size_t stage_id = 0; stage_id < dag->ops.size(); stage_id++) {
    if (is_strictly_inlinable.count(stage_id)) {
      ICHECK(dag->access_analyzer.IsStrictlyInlineable(dag->ops[stage_id]));
    } else {
      ICHECK(!dag->access_analyzer.IsStrictlyInlineable(dag->ops[stage_id]));
    }
  }

  std::set<int> is_output = {relu};
  for (size_t stage_id = 0; stage_id < dag->ops.size(); stage_id++) {
    if (is_output.count(stage_id)) {
      ICHECK(dag->access_analyzer.IsOutput(dag->ops[stage_id]));
    } else {
      ICHECK(!dag->access_analyzer.IsOutput(dag->ops[stage_id]));
    }
  }

  ICHECK_EQ(dag->access_analyzer.GetNumCommonOuterIterator(dag->ops[conv], dag->ops[bias_add]), 4);
  ICHECK_EQ(dag->access_analyzer.GetNumCommonOuterIterator(dag->ops[conv], dag->ops[relu]), 4);
  ICHECK_EQ(dag->access_analyzer.GetNumCommonOuterIterator(dag->ops[data], dag->ops[relu]), 1);

  ICHECK(dag->access_analyzer.ElementWiseMatch(dag->ops[conv], dag->ops[bias_add]));
  ICHECK(dag->access_analyzer.ElementWiseMatch(dag->ops[conv], dag->ops[relu]));
  ICHECK(!dag->access_analyzer.ElementWiseMatch(dag->ops[data], dag->ops[padding]));

  std::unordered_set<tvm::te::Operation, tvm::ObjectHash, tvm::ObjectEqual> op_set;
  {
    std::vector<std::pair<int, int>> consumer_list = {
        {data, padding},     {padding, conv},    {kernel, conv},     {conv, bias_add},
        {bias, bias_add},    {bias_add, bn_mul}, {bn_scale, bn_mul}, {bn_mul, bn_add},
        {bn_offset, bn_add}, {bn_add, relu}};
    for (const auto& pair : consumer_list) {
      op_set = dag->access_analyzer.GetConsumers(s0, s0->stages[pair.first]->op);
      ICHECK_EQ(op_set.size(), 1);
      ICHECK_EQ((*op_set.begin()), s0->stages[pair.second]->op);
    }
    std::vector<std::pair<int, std::vector<int>>> producer_list = {{padding, {data}},
                                                                   {conv, {padding, kernel}},
                                                                   {bias_add, {conv, bias}},
                                                                   {bn_mul, {bias_add, bn_scale}},
                                                                   {bn_add, {bn_mul, bn_offset}},
                                                                   {relu, {bn_add}}};
    for (const auto& pair : producer_list) {
      op_set = dag->access_analyzer.GetProducers(s0, s0->stages[pair.first]->op);
      ICHECK_EQ(op_set.size(), pair.second.size());
      for (const auto& target : pair.second) {
        ICHECK(op_set.count(s0->stages[target]->op));
      }
    }
  }

  s0.compute_inline(bn_add);
  s0.compute_inline(bn_mul);
  s0.compute_inline(bias_add);
  s0.compute_inline(padding);
  {
    std::vector<std::pair<int, int>> consumer_list = {{data, conv}, {kernel, conv}, {conv, relu}};
    for (const auto& pair : consumer_list) {
      op_set = dag->access_analyzer.GetConsumers(s0, s0->stages[pair.first]->op);
      ICHECK_EQ(op_set.size(), 1);
      ICHECK_EQ((*op_set.begin()), s0->stages[pair.second]->op);
    }
    std::vector<std::pair<int, std::vector<int>>> producer_list = {{padding, {data}},
                                                                   {conv, {padding, kernel}},
                                                                   {bias_add, {conv, bias}},
                                                                   {bn_mul, {bias_add, bn_scale}},
                                                                   {bn_add, {bn_mul, bn_offset}},
                                                                   {relu, {bn_add}}};
    for (const auto& pair : producer_list) {
      op_set = dag->access_analyzer.GetDirectProducers(s0->stages[pair.first]->op);
      ICHECK_EQ(op_set.size(), pair.second.size());
      for (const auto& target : pair.second) {
        ICHECK(op_set.count(s0->stages[target]->op));
      }
    }
  }
}
