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
#include <tvm/te/operation.h>

#include <cmath>
#include <string>

TEST(Runtime, ZeroCopy) {
  /*
   *
   *          A    B
   *           \  /
   *      elemwise_add(out0)
   *              \
   *       C      copy
   *        \      /
   *      elemwise_sub(out1)
   */

  using namespace tvm;
  using namespace tvm::te;

  auto target_llvm = Target("llvm");

  // The shape of input tensors.
  const int n = 4;
  Array<PrimExpr> shape{n};

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");
  auto C = placeholder(shape, DataType::Float(32), "C");

  auto elemwise_add = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "elemwise_add");

  auto copy = placeholder(shape, DataType::Float(32), "__copy");
  auto elemwise_sub = compute(
      C->shape, [&copy, &C](PrimExpr i) { return copy[i] - C[i]; }, "elemwise_sub");

  With<Target> llvm_scope(target_llvm);
  auto s1 = create_schedule({elemwise_add->op});
  auto s2 = create_schedule({elemwise_sub->op});

  auto args1 = Array<Tensor>({A, B, elemwise_add});
  auto args2 = Array<Tensor>({copy, C, elemwise_sub});

  std::unordered_map<Tensor, Buffer> binds;
  auto lowered_s1 = LowerSchedule(s1, args1, "elemwise_add", binds);
  auto lowered_s2 = LowerSchedule(s2, args2, "elemwise_sub", binds);
  Map<tvm::Target, IRModule> inputs = {{target_llvm, lowered_s1}, {target_llvm, lowered_s2}};
  auto module = build(inputs, Target());

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
      "\"heads\": [[2, 0, 0], [5, 0, 0]], \"attrs\": {\"storage_id\": [\"list_int\", "
      "[3, 4, 0, 1, 5, 2]], \"shape\": [\"list_shape\", [[4], [4], [4], [4], [4], "
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

  // // Initialize graph executor.
  int device_type = static_cast<int>(kDLCPU);
  int device_id = 0;

  const runtime::PackedFunc* graph_executor =
      tvm::runtime::Registry::Get("tvm.graph_executor.create");
  runtime::Module mod = (*graph_executor)(json, module, device_type, device_id);

  // test FFI for module.
  auto test_ffi = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
    int tcode = args[1];
    ICHECK_EQ(args[0].type_code(), tcode);
  });

  test_ffi(runtime::Module(mod), static_cast<int>(kTVMModuleHandle));
  test_ffi(Optional<runtime::Module>(mod), static_cast<int>(kTVMModuleHandle));

  PackedFunc set_input_zero_copy = mod.GetFunction("set_input_zero_copy", false);
  PackedFunc set_output_zero_copy = mod.GetFunction("set_output_zero_copy", false);
  PackedFunc run = mod.GetFunction("run", false);
  set_input_zero_copy("A", a_val);
  set_input_zero_copy("B", b_val);
  set_input_zero_copy("C", c_val);

  tvm::runtime::NDArray out0 = runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  tvm::runtime::NDArray out1 = runtime::NDArray::Empty({n}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  set_output_zero_copy("elemwise_add", out0);
  set_output_zero_copy("elemwise_sub", out1);

  run();
  auto p_out0 = static_cast<float*>(out0->data);
  auto p_out1 = static_cast<float*>(out1->data);

  // Check correctness.
  for (int i = 0; i < n; ++i) {
    ICHECK_LT(std::fabs(p_out0[i] - (i + (i + 1.0))), 1e-5);
  }

  for (int i = 0; i < n; ++i) {
    ICHECK_LT(std::fabs(p_out1[i] - (i + (i + 1.0) - (i - 1.0))), 1e-5);
  }
}
