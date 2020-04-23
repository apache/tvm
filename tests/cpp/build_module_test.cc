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
#include <topi/cuda/injective.h>
#include <tvm/te/operation.h>
#include <tvm/runtime/registry.h>
#include <tvm/driver/driver_api.h>

#include <string>
#include <cmath>

TEST(BuildModule, Basic) {
  using namespace tvm;
  using namespace tvm::te;
  auto n = var("n");
  Array<PrimExpr> shape;
  shape.push_back(n);

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");

  auto C = compute(A->shape, [&A, &B](PrimExpr i) {
    return A[i] + B[i];
  }, "C");

  auto s = create_schedule({ C->op });

  auto cAxis = C->op.as<ComputeOpNode>()->axis;

  IterVar bx, tx;
  s[C].split(cAxis[0], 64, &bx, &tx);

  auto args = Array<Tensor>({ A, B, C });
  std::unordered_map<Tensor, Buffer> binds;

  auto config = BuildConfig::Create();
  auto target = target::llvm();

  auto lowered = lower(s, args, "func", binds, config);
  auto module = build(lowered, target, Target(), config);

  auto mali_target = Target::Create("opencl -model=Mali-T860MP4@800Mhz -device=mali");
  CHECK_EQ(mali_target->str(), "opencl -model=Mali-T860MP4@800Mhz -device=mali");
}

TEST(BuildModule, Heterogeneous) {
  /* The testing network is like following, where the element-wise add and sub
   * ops are allocated to GPU and CPU, respectively:
   *
   *          A    B
   *           \  /
   *      elemwise_add  (gpu)
   *              \
   *              copy      C
   *                \      /
   *              elemwise_sub  (cpu)
   */

  using namespace tvm;
  using namespace tvm::te;
  bool enabled = tvm::runtime::RuntimeEnabled("cuda");
  if (!enabled) {
    LOG(INFO) << "Skip heterogeneous test because cuda is not enabled."
              << "\n";
    return;
  }

  auto target_llvm = target::llvm();
  auto target_cuda = target::cuda();

  // The shape of input tensors.
  const int n = 4;
  Array<PrimExpr> shape{n};

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");
  auto C = placeholder(shape, DataType::Float(32), "C");

  auto elemwise_add = compute(A->shape, [&A, &B](PrimExpr i) {
    return A[i] + B[i];
  }, "elemwise_add");

  auto copy = placeholder(shape, DataType::Float(32), "__copy");
  auto elemwise_sub = compute(C->shape, [&copy, &C](PrimExpr i) {
    return copy[i] - C[i];
  }, "elemwise_sub");

  With<Target> cuda_scope(target_cuda);
  auto s1 = topi::cuda::schedule_injective(target_cuda, {elemwise_add});


  With<Target> llvm_scope(target_llvm);
  auto s2 = create_schedule({elemwise_sub->op});

  auto config = BuildConfig::Create();
  auto args1 = Array<Tensor>({A, B, elemwise_add});
  auto args2 = Array<Tensor>({copy, C, elemwise_sub});

  std::unordered_map<Tensor, Buffer> binds;
  auto lowered_s1 = lower(s1, args1, "elemwise_add", binds, config);
  auto lowered_s2 = lower(s2, args2, "elemwise_sub", binds, config);
  Map<tvm::Target, IRModule> inputs = {{target_cuda, lowered_s1},
                                       {target_llvm, lowered_s2}};
  auto module = build(inputs, Target(), config);

  // Assertion for build.
  CHECK_EQ(module->imports().size(), 1);

  // Execute the graph and check the correctness.
  // Setup graph json.
  std::string json =
      "{\"nodes\": [{\"op\": \"null\", \"name\": \"A\", \"inputs\": []}, "
      "{\"op\": \"null\", \"name\": \"B\", \"inputs\": []}, {\"op\": "
      "\"tvm_op\", \"name\": \"elemwise_add\", \"attrs\": {\"flatten_data\": "
      "\"1\", \"func_name\": \"elemwise_add\", \"num_inputs\": \"2\", "
      "\"num_outputs\": \"1\"}, \"inputs\": [[0, 0, 0], [1, 0, 0]]}, {\"op\": "
      "\"tvm_op\", \"name\": \"__copy_add_to_sub\", \"attrs\": "
      "{\"flatten_data\": \"0\", \"func_name\": \"__copy\", \"num_inputs\": "
      "\"1\", \"num_outputs\": \"1\"}, \"inputs\": [[2, 0, 0]]}, {\"op\": "
      "\"null\", \"name\": \"C\", \"inputs\": []}, {\"op\": \"tvm_op\", "
      "\"name\": \"elemwise_sub\", \"attrs\": {\"flatten_data\": \"0\", "
      "\"func_name\": \"elemwise_sub\", \"num_inputs\": \"2\", "
      "\"num_outputs\": \"1\"}, \"inputs\": [[3, 0, 0], [4, 0, 0]]}], "
      "\"arg_nodes\": [0, 1, 4], \"node_row_ptr\": [0, 1, 2, 3, 4, 5, 6], "
      "\"heads\": [[5, 0, 0]], \"attrs\": {\"storage_id\": [\"list_int\", [3, "
      "4, 0, 1, 5, 2]], \"shape\": [\"list_shape\", [[4], [4], [4], [4], [4], "
      "[4]]], \"device_index\": [\"list_int\", [2, 2, 2, 1, 1, 1]], \"dtype\": "
      "[\"list_int\", [0, 0, 0, 0, 0, 0]], \"dltype\": [\"list_str\", "
      "[\"float32\", \"float32\", \"float32\", \"float32\", \"float32\", "
      "\"float32\"]]}}";

  // Setup inputs.
  auto a_val =
      runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto b_val =
      runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto c_val =
      runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  auto pa = (float*)(a_val->data);
  auto pb = (float*)(b_val->data);
  auto pc = (float*)(c_val->data);

  // Assign values.
  for (int i = 0; i < n; i++) {
    pa[i] = i;
    pb[i] = i + 1.0;
    pc[i] = i - 1.0;
  }

  // Initialize graph runtime.
  int cpu_dev_ty = static_cast<int>(kDLCPU);
  int cpu_dev_id = 0;
  int gpu_dev_ty = static_cast<int>(kDLGPU);
  int gpu_dev_id = 0;

  const runtime::PackedFunc* graph_runtime =
      tvm::runtime::Registry::Get("tvm.graph_runtime.create");
  runtime::Module mod = (*graph_runtime)(
      json, module, cpu_dev_ty, cpu_dev_id, gpu_dev_ty, gpu_dev_id);

  PackedFunc set_input = mod.GetFunction("set_input", false);
  PackedFunc run = mod.GetFunction("run", false);
  PackedFunc get_output = mod.GetFunction("get_output", false);
  set_input("A", a_val);
  set_input("B", b_val);
  set_input("C", c_val);

  run();
  tvm::runtime::NDArray out = get_output(0);
  float* p_out = (float*)out->data;

  // Check correctness.
  for (int i = 0; i < n; ++i) {
    CHECK_LT(std::fabs(p_out[i] - (i + (i + 1.0) - (i - 1.0))), 1e-5);
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
