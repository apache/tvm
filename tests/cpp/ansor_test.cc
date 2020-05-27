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
#include <unordered_set>
#include <tvm/te/operation.h>
#include <topi/nn.h>
#include "../../src/ansor/loop_state.h"

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

  int OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) / strides + 1;
  int OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) / strides + 1;

  const auto& conv = topi::conv2d_nchw(data, kernel, padding, padding, strides,
                                       strides);
  CHECK(conv->shape[2].as<IntImmNode>()->value == OH);
  CHECK(conv->shape[3].as<IntImmNode>()->value == OW);

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
  const auto& dag = tvm::ansor::ComputeDAGNode::make(tensors);
  const auto& state = tvm::ansor::StateNode::make(dag->ops);
  CHECK(std::equal_to<tvm::ansor::State>()(state, dag.GetInitState()));

  LOG(INFO) << "\n" << state;
  LOG(INFO) << "\n" << dag;
  LOG(INFO) << "\n" << dag->access_analyzer;
}

TEST(ComputeDAG, GetProducersConsumers) {
  using namespace tvm::ansor;

  const auto& tensors = conv2d_nchw_bn_relu_func(1, 224, 224, 3, 64, 7, 2, 3);
  const auto& dag = tvm::ansor::ComputeDAGNode::make(tensors);
  int data = 0, padding = 1, kernel = 2, conv = 3, bias = 4, bias_add = 5;
  int bn_scale = 6, bn_mul = 7, bn_offset = 8, bn_add = 9, relu = 10;

  State s0 = dag.GetInitState();
  std::unordered_set<tvm::te::Operation, tvm::ObjectHash, tvm::ObjectEqual> set;
  {
    std::vector<std::pair<int, int>> consumer_list = {
      {data, padding}, {padding, conv}, {kernel, conv}, {conv, bias_add},
      {bias, bias_add}, {bias_add, bn_mul}, {bn_scale, bn_mul},
      {bn_mul, bn_add}, {bn_offset, bn_add}, {bn_add, relu}
    };
    for (const auto& pair : consumer_list) {
      dag->access_analyzer.GetConsumers(s0, s0->stages[pair.first]->op, &set);
      CHECK_EQ(set.size(), 1);
      CHECK_EQ((*set.begin()), s0->stages[pair.second]->op);
    }
    std::vector<std::pair<int, std::vector<int>>> producer_list = {
      {padding, {data}}, {conv, {padding, kernel}}, {bias_add, {conv, bias}},
      {bn_mul, {bias_add, bn_scale}}, {bn_add, {bn_mul, bn_offset}},
      {relu, {bn_add}}
    };
    for (const auto& pair : producer_list) {
      dag->access_analyzer.GetProducers(s0, s0->stages[pair.first]->op, &set);
      CHECK_EQ(set.size(), pair.second.size());
      for (const auto& target : pair.second) {
        CHECK(set.count(s0->stages[target]->op));
      }
    }
  }

  s0.compute_inline(bn_add);
  s0.compute_inline(bn_mul);
  s0.compute_inline(bias_add);
  s0.compute_inline(padding);
  {
    std::vector<std::pair<int, int>> consumer_list = {
      {data, conv}, {kernel, conv}, {conv, relu}
    };
    for (const auto& pair : consumer_list) {
      dag->access_analyzer.GetConsumers(s0, s0->stages[pair.first]->op, &set);
      CHECK_EQ(set.size(), 1);
      CHECK_EQ((*set.begin()), s0->stages[pair.second]->op);
    }
    std::vector<std::pair<int, std::vector<int>>> producer_list = {
      {padding, {data}}, {conv, {padding, kernel}}, {bias_add, {conv, bias}},
      {bn_mul, {bias_add, bn_scale}}, {bn_add, {bn_mul, bn_offset}},
      {relu, {bn_add}}
    };
    for (const auto& pair : producer_list) {
      dag->access_analyzer.GetProducers(s0, s0->stages[pair.first]->op, &set);
      CHECK_EQ(set.size(), pair.second.size());
      for (const auto& target : pair.second) {
        CHECK(set.count(s0->stages[target]->op));
      }
    }
  }
}

TEST(Step, SplitFuseReorder) {
  using namespace tvm::ansor;

  const auto& tensors = matmul_func(512, 512, 512);
  const auto& dag = ComputeDAGNode::make(tensors);

  State s0 = dag.GetInitState();
  State s1 = s0;
  Iterator ti = s0->stages[2]->iters[0];
  Iterator tj = s0->stages[2]->iters[1];
  Iterator tk = s0->stages[2]->iters[2];
  std::vector<Iterator> its;

  CHECK_EQ(s1->stages[2]->iters[0]->range->extent.as<IntImmNode>()->value, 512);

  its = s0.split(2, ti, {16});
  CHECK_EQ(s0->stages[2]->iters[0]->range->extent.as<IntImmNode>()->value, 32);
  CHECK_EQ(s0->stages[2]->iters[1]->range->extent.as<IntImmNode>()->value, 16);

  Iterator tio = its[0], tii = its[1];
  its = s0.split(2, tj, {8});
  CHECK_EQ(s0->stages[2]->iters[2]->range->extent.as<IntImmNode>()->value, 64);
  CHECK_EQ(s0->stages[2]->iters[3]->range->extent.as<IntImmNode>()->value, 8);

  Iterator tjo = its[0], tji = its[1];
  s0.reorder(2, {tio, tjo, tk, tji, tii});
  CHECK_EQ(s0->stages[2]->iters[2]->range->extent.as<IntImmNode>()->value, 512);

  s0.fuse(2, {tio, tjo});
  CHECK_EQ(s0->stages[2]->iters[0]->range->extent.as<IntImmNode>()->value, 2048);

  s1.split(2, ti, {8, 2});
  s1.split(2, tj, {32, 8}, false);
  CHECK_EQ(s1->stages[2]->iters[0]->range->extent.as<IntImmNode>()->value, 32);
  CHECK_EQ(s1->stages[2]->iters[1]->range->extent.as<IntImmNode>()->value, 8);
  CHECK_EQ(s1->stages[2]->iters[2]->range->extent.as<IntImmNode>()->value, 2);
  CHECK_EQ(s1->stages[2]->iters[3]->range->extent.as<IntImmNode>()->value, 32);
  CHECK_EQ(s1->stages[2]->iters[4]->range->extent.as<IntImmNode>()->value, 8);
  CHECK_EQ(s1->stages[2]->iters[5]->range->extent.as<IntImmNode>()->value, 2);
}

TEST(Step, ComputeAtRootInline) {
  using namespace tvm::ansor;

  const auto& tensors = conv2d_nchw_bn_relu_func(1, 224, 224, 3, 64, 7, 2, 3);
  const auto& dag = tvm::ansor::ComputeDAGNode::make(tensors);
  // int data = 0, padding = 1, kernel = 2;
  int conv = 3;
  // int bias = 4;
  int bias_add = 5;
  // int bn_scale = 6;
  int bn_mul = 7;
  // int bn_offset = 8;
  int bn_add = 9, relu = 10;

  State s0 = dag.GetInitState();
  s0.compute_inline(bn_add);
  s0.compute_inline(bn_mul);
  s0.compute_inline(bias_add);
  s0.compute_at(conv, relu, s0->stages[relu]->iters[2]);
  const auto& conv_stage_attach = s0->attach_map->stage_to_attach_iter.find(conv);
  std::pair<int, int> iterkey(relu, 2);
  CHECK(conv_stage_attach->second == iterkey);
  const auto& conv_iter_attach = s0->attach_map->iter_to_attached_stages.find(iterkey);
  CHECK_EQ(conv_iter_attach->second.size(), 1);
  CHECK_EQ(conv_iter_attach->second[0], conv);
  std::stringstream ss;
  ss << "Placeholder: Data, Kernel, Bias, Bn_scale, Bn_offset\n"
     << "for ax1 (0,3)\n"
     << "  for ax2 (0,230)\n"
     << "    for ax3 (0,230)\n"
     << "      T_pad = ...\n"
     << "for ax1 (0,64)\n"
     << "  for ax2 (0,112)\n"
     << "    for ax0 (None)\n"
     << "      for ax1 (None)\n"
     << "        for ax2 (None)\n"
     << "          for ax3 (None)\n"
     << "            for i (None)\n"
     << "              for kh (None)\n"
     << "                for kw (None)\n"
     << "                  T_conv2d_nchw = ...\n"
     << "    for ax3 (0,112)\n"
     << "      T_relu = ...\n";
  CHECK_EQ(s0.ToStr().compare(ss.str()), 0);

  s0.compute_root(conv);
  s0.compute_root(bn_mul);
  CHECK_EQ(s0->attach_map->stage_to_attach_iter.size(), 0);
  CHECK_EQ(s0->attach_map->iter_to_attached_stages.size(), 0);
  ss.str(std::string());
  ss << "Placeholder: Data, Kernel, Bias, Bn_scale, Bn_offset\n"
     << "for ax1 (0,3)\n"
     << "  for ax2 (0,230)\n"
     << "    for ax3 (0,230)\n"
     << "      T_pad = ...\n"
     << "for ax0 (None)\n"
     << "  for ax1 (None)\n"
     << "    for ax2 (None)\n"
     << "      for ax3 (None)\n"
     << "        for i (None)\n"
     << "          for kh (None)\n"
     << "            for kw (None)\n"
     << "              T_conv2d_nchw = ...\n"
     << "for ax0 (None)\n"
     << "  for ax1 (None)\n"
     << "    for ax2 (None)\n"
     << "      for ax3 (None)\n"
     << "        Bn_mul = ...\n"
     << "for ax1 (0,64)\n"
     << "  for ax2 (0,112)\n"
     << "    for ax3 (0,112)\n"
     << "      T_relu = ...\n";
  CHECK_EQ(s0.ToStr().compare(ss.str()), 0);
}

TEST(Step, CacheReadWrite) {
  using namespace tvm;
  using namespace tvm::te;
  using namespace tvm::ansor;

  const auto& test_func = []() -> Array<Tensor> {
    int N = 4, H = 7, W = 7, CO = 512, CI = 512, KH = 3, KW = 3, stride = 1;
    int padding = 1;
    Tensor data = placeholder({N, CI, H, W}, DataType::Float(32), "Data");
    Tensor kernel_data = placeholder({CO, CI, KH, KW}, DataType::Float(32),
                                    "kernel_data");
    const auto& k_split = compute(kernel_data->shape,
        [&](const Array<Var>& i) {
            return Array<PrimExpr>({kernel_data[i[0]][i[1]][i[2]][i[3]] + 1,
                div(kernel_data[i[0]][i[1]][i[2]][i[3]], 2)});
        },
        "Kernel_split");
    const auto& kernel = compute(kernel_data->shape,
        [&](Var i, Var j, Var k, Var l) {
            return (k_split[0])[i][j][k][l] + (k_split[1])[i][j][k][l];
        },
        "Kernel");
    const auto& conv = topi::conv2d_nchw(data, kernel, padding, padding, stride,
                                        stride);
    const auto& relu = topi::relu<float>(conv);
    const auto& out = compute(relu->shape,
        [&](Var i, Var j, Var k, Var l) {
            return data[i][j][k][l] + relu[i][j][k][l];
        },
        "Add");
    return {data, kernel_data, out};
  };
  const auto& dag = ComputeDAGNode::make(test_func());

  int data = 0, pad_temp = 1, kernel_data = 2, kernel_split = 3, kernel = 4;
  int conv = 5, relu = 6, add = 7;

  // 0: init state
  auto s0 = dag.GetInitState();
  std::vector<Iterator> ori_its = s0->stages[add]->iters;
  auto its = s0.split(add, s0->stages[add]->iters[0], {2});
  s0.reorder(add, {its[0], ori_its[1], its[1], ori_its[2], ori_its[3]});
  s0.compute_inline(relu);

  // 1: simple cache_write with compute_at
  int conv_global = s0.cache_write(conv, "global", dag);
  conv++; relu++; add++;
  s0.compute_at(conv_global, conv, s0->stages[conv]->iters[3]);

  // 2: simple cache_read with compute_at
  int kernel_global = s0.cache_read(kernel, "global", {conv_global}, dag);
  conv_global++; conv++; relu++; add++;
  s0.compute_at(kernel_global, conv_global, s0->stages[conv_global]->iters[4]);
  std::stringstream ss;
  ss << "Placeholder: Data, kernel_data\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,9)\n"
     << "      for ax3 (0,9)\n"
     << "        T_pad = ...\n"
     << "for ax0 (0,512)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,3)\n"
     << "      for ax3 (0,3)\n"
     << "        Kernel_split = ...\n"
     << "for ax0 (0,512)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,3)\n"
     << "      for ax3 (0,3)\n"
     << "        Kernel = ...\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,7)\n"
     << "      for ax3 (0,7)\n"
     << "        for ax0_c (None)\n"
     << "          for ax1_c (None)\n"
     << "            for ax2_c (None)\n"
     << "              for ax3_c (None)\n"
     << "                for i (None)\n"
     << "                  for ax0 (None)\n"
     << "                    for ax1 (None)\n"
     << "                      for ax2 (None)\n"
     << "                        for ax3 (None)\n"
     << "                          Kernel.global = ...\n"
     << "                  for kh (None)\n"
     << "                    for kw (None)\n"
     << "                      T_conv2d_nchw.global = ...\n"
     << "        T_conv2d_nchw = ...\n"
     << "for ax0.0 (0,2)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax0.1 (0,2)\n"
     << "      for ax2 (0,7)\n"
     << "        for ax3 (0,7)\n"
     << "          Add = ...\n";
  CHECK_EQ(s0.ToStr().compare(ss.str()), 0);

  // 3: two level cache_read with compute_at
  //    preparing for GPU's shared memory & local memory
  int pad_temp_global = s0.cache_read(pad_temp, "global", {conv_global}, dag);
  kernel_data++; kernel_split++; kernel++; kernel_global++;
  conv_global++; conv++; relu++; add++;
  int pad_temp_shared = s0.cache_read(pad_temp_global, "shared", {conv_global},
                                      dag);
  kernel_data++; kernel_split++; kernel++; kernel_global++;
  conv_global++; conv++; relu++; add++;
  s0.compute_at(pad_temp_global, conv_global,
                s0->stages[conv_global]->iters[2]);
  s0.compute_at(pad_temp_shared, conv_global,
                s0->stages[conv_global]->iters[4]);

  // 4: cache_read with multi readers
  // This stage cannot be compute at to its consumer
  s0.cache_read(data, "global", {pad_temp, add}, dag);
  pad_temp++; pad_temp_global++; pad_temp_shared++;
  kernel_data++; kernel_split++; kernel++; kernel_global++;
  conv_global++; conv++; relu++; add++;
  ss.str(std::string());
  ss << "Placeholder: Data, kernel_data\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,7)\n"
     << "      for ax3 (0,7)\n"
     << "        Data.global = ...\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,9)\n"
     << "      for ax3 (0,9)\n"
     << "        T_pad = ...\n"
     << "for ax0 (0,512)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,3)\n"
     << "      for ax3 (0,3)\n"
     << "        Kernel_split = ...\n"
     << "for ax0 (0,512)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,3)\n"
     << "      for ax3 (0,3)\n"
     << "        Kernel = ...\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,7)\n"
     << "      for ax3 (0,7)\n"
     << "        for ax0_c (None)\n"
     << "          for ax1_c (None)\n"
     << "            for ax2_c (None)\n"
     << "              for ax0 (None)\n"
     << "                for ax1 (None)\n"
     << "                  for ax2 (None)\n"
     << "                    for ax3 (None)\n"
     << "                      T_pad.global = ...\n"
     << "              for ax3_c (None)\n"
     << "                for i (None)\n"
     << "                  for ax0 (None)\n"
     << "                    for ax1 (None)\n"
     << "                      for ax2 (None)\n"
     << "                        for ax3 (None)\n"
     << "                          Kernel.global = ...\n"
     << "                  for ax0 (None)\n"
     << "                    for ax1 (None)\n"
     << "                      for ax2 (None)\n"
     << "                        for ax3 (None)\n"
     << "                          T_pad.global.shared = ...\n"
     << "                  for kh (None)\n"
     << "                    for kw (None)\n"
     << "                      T_conv2d_nchw.global = ...\n"
     << "        T_conv2d_nchw = ...\n"
     << "for ax0.0 (0,2)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax0.1 (0,2)\n"
     << "      for ax2 (0,7)\n"
     << "        for ax3 (0,7)\n"
     << "          Add = ...\n";
  CHECK_EQ(s0.ToStr().compare(ss.str()), 0);

  // 5: cache_write with multi outputs
  // TVM's cache_write actually has a bug with this case:

  // After schedule.cache_write, TVM generate one new stage:
  //   From: kernel_data -> kernel_split -> kernel
  //   To:   kernel_data -> kernel_split_global -> kernel_split -> kernel

  // But with topo sort analyse, we get:
  //   kernel_data -> kernel_split_global -> kernel_split -> kernel
  //         \                                                /
  //          ----------------> kernel_split ---------------->

  // Seems there's bug with the input/output tensor. Such multi outputs case
  // should be unusual, so we make some hack on DoCacheWrite
  // To be fixed in the future
  s0.cache_write(kernel_split, "global", dag);
  ss.str(std::string());
  ss << "Placeholder: Data, kernel_data\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,7)\n"
     << "      for ax3 (0,7)\n"
     << "        Data.global = ...\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,9)\n"
     << "      for ax3 (0,9)\n"
     << "        T_pad = ...\n"
     << "for ax0_c (0,512)\n"
     << "  for ax1_c (0,512)\n"
     << "    for ax2_c (0,3)\n"
     << "      for ax3_c (0,3)\n"
     << "        Kernel_split.global = ...\n"
     << "for ax0 (0,512)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,3)\n"
     << "      for ax3 (0,3)\n"
     << "        Kernel_split = ...\n"
     << "for ax0 (0,512)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,3)\n"
     << "      for ax3 (0,3)\n"
     << "        Kernel_split = ...\n"
     << "for ax0 (0,512)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,3)\n"
     << "      for ax3 (0,3)\n"
     << "        Kernel = ...\n"
     << "for ax0 (0,4)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax2 (0,7)\n"
     << "      for ax3 (0,7)\n"
     << "        for ax0_c (None)\n"
     << "          for ax1_c (None)\n"
     << "            for ax2_c (None)\n"
     << "              for ax0 (None)\n"
     << "                for ax1 (None)\n"
     << "                  for ax2 (None)\n"
     << "                    for ax3 (None)\n"
     << "                      T_pad.global = ...\n"
     << "              for ax3_c (None)\n"
     << "                for i (None)\n"
     << "                  for ax0 (None)\n"
     << "                    for ax1 (None)\n"
     << "                      for ax2 (None)\n"
     << "                        for ax3 (None)\n"
     << "                          Kernel.global = ...\n"
     << "                  for ax0 (None)\n"
     << "                    for ax1 (None)\n"
     << "                      for ax2 (None)\n"
     << "                        for ax3 (None)\n"
     << "                          T_pad.global.shared = ...\n"
     << "                  for kh (None)\n"
     << "                    for kw (None)\n"
     << "                      T_conv2d_nchw.global = ...\n"
     << "        T_conv2d_nchw = ...\n"
     << "for ax0.0 (0,2)\n"
     << "  for ax1 (0,512)\n"
     << "    for ax0.1 (0,2)\n"
     << "      for ax2 (0,7)\n"
     << "        for ax3 (0,7)\n"
     << "          Add = ...\n";
  CHECK_EQ(s0.ToStr().compare(ss.str()), 0);
}

TEST(Step, FollowSplitFollowFusedSplit) {
  // todo
}

TEST(Step, Rfactor) {
  // todo
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
