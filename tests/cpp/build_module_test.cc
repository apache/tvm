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
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/topi/cuda/injective.h>

#include <cmath>
#include <string>

TEST(BuildModule, Basic) {
  using namespace tvm;
  using namespace tvm::te;
  auto n = var("n");
  Array<PrimExpr> shape;
  shape.push_back(n);

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");

  auto C = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "C");

  auto s = create_schedule({C->op});

  auto cAxis = C->op.as<ComputeOpNode>()->axis;

  IterVar bx, tx;
  s[C].split(cAxis[0], 64, &bx, &tx);

  auto args = Array<Tensor>({A, B, C});
  std::unordered_map<Tensor, Buffer> binds;

  auto target = Target("llvm");

  auto lowered = LowerSchedule(s, args, "func", binds, GlobalVarSupply());
  auto module = build(lowered, target, Target());

  auto mali_target = Target("opencl -model=Mali-T860MP4@800Mhz -device=mali");
  ICHECK_EQ(mali_target->kind->name, "opencl");
  ICHECK_EQ(mali_target->keys.size(), 3);
  ICHECK_EQ(mali_target->keys[0], "mali");
  ICHECK_EQ(mali_target->keys[1], "opencl");
  ICHECK_EQ(mali_target->keys[2], "gpu");
  ICHECK_EQ(mali_target->GetAttr<String>("device").value(), "mali");
  ICHECK_EQ(mali_target->GetAttr<String>("model").value(), "Mali-T860MP4@800Mhz");
  ICHECK_EQ(mali_target->GetAttr<Integer>("max_num_threads").value(), 256);
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

  auto target_llvm = Target("llvm");
  auto target_cuda = Target("cuda");

  // The shape of input tensors.
  const int n = 4;
  Array<PrimExpr> shape{n};

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");
  auto C = placeholder(shape, DataType::Float(32), "C");

  auto elemwise_add = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "elemwise_add");

  // TODO(mbs): device_copy cleanup.
  auto copy = placeholder(shape, DataType::Float(32), "__copy");
  auto elemwise_sub = compute(
      C->shape, [&copy, &C](PrimExpr i) { return copy[i] - C[i]; }, "elemwise_sub");

  auto fcreate_s1 = [=]() {
    With<Target> cuda_scope(target_cuda);
    return topi::cuda::schedule_injective(target_cuda, {elemwise_add});
  };

  auto fcreate_s2 = [=]() {
    With<Target> llvm_scope(target_llvm);
    return create_schedule({elemwise_sub->op});
  };

  auto args1 = Array<Tensor>({A, B, elemwise_add});
  auto args2 = Array<Tensor>({copy, C, elemwise_sub});

  std::unordered_map<Tensor, Buffer> binds;
  GlobalVarSupply global_var_supply = GlobalVarSupply();
  auto lowered_s1 = LowerSchedule(fcreate_s1(), args1, "elemwise_add", binds, global_var_supply);
  auto lowered_s2 = LowerSchedule(fcreate_s2(), args2, "elemwise_sub", binds, global_var_supply);
  Map<tvm::Target, IRModule> inputs = {{target_cuda, lowered_s1}, {target_llvm, lowered_s2}};
  auto module = build(inputs, Target());

  // Assertion for build.
  ICHECK_EQ(module->imports().size(), 1);

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
  auto a_val = runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto b_val = runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto c_val = runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  auto pa = static_cast<float*>(a_val->data);
  auto pb = static_cast<float*>(b_val->data);
  auto pc = static_cast<float*>(c_val->data);

  // Assign values.
  for (int i = 0; i < n; i++) {
    pa[i] = i;
    pb[i] = i + 1.0;
    pc[i] = i - 1.0;
  }

  // Initialize graph executor.
  int cpu_dev_ty = static_cast<int>(kDLCPU);
  int cpu_dev_id = 0;
  int gpu_dev_ty = static_cast<int>(kDLCUDA);
  int gpu_dev_id = 0;

  const runtime::PackedFunc* graph_executor =
      tvm::runtime::Registry::Get("tvm.graph_executor.create");
  runtime::Module mod =
      (*graph_executor)(json, module, cpu_dev_ty, cpu_dev_id, gpu_dev_ty, gpu_dev_id);

  // test FFI for module.
  auto test_ffi = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
    int tcode = args[1];
    ICHECK_EQ(args[0].type_code(), tcode);
  });

  test_ffi(runtime::Module(mod), static_cast<int>(kTVMModuleHandle));
  test_ffi(Optional<runtime::Module>(mod), static_cast<int>(kTVMModuleHandle));

  PackedFunc set_input = mod.GetFunction("set_input", false);
  PackedFunc run = mod.GetFunction("run", false);
  PackedFunc get_output = mod.GetFunction("get_output", false);
  set_input("A", a_val);
  set_input("B", b_val);
  set_input("C", c_val);

  run();
  tvm::runtime::NDArray out = get_output(0);
  float* p_out = static_cast<float*>(out->data);

  // Check correctness.
  for (int i = 0; i < n; ++i) {
    ICHECK_LT(std::fabs(p_out[i] - (i + (i + 1.0) - (i - 1.0))), 1e-5);
  }
}
