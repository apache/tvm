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
#include <tvm/build_module.h>
#include <tvm/operation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <topi/generic/injective.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>


TVM_REGISTER_GLOBAL("test.sch")
.set_body([](tvm::TVMArgs args, tvm::TVMRetValue *rv) {
  *rv = topi::generic::schedule_injective(args[0], args[1]);
  });

TEST(Relay, BuildModule) {
  using namespace tvm;
  auto tensor_type = relay::TensorTypeNode::make({2, 3}, ::tvm::Float(32));
  auto a = relay::VarNode::make("a", tensor_type);
  auto b = relay::VarNode::make("b", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto x = relay::CallNode::make(add_op, {a, b}, tvm::Attrs(), {});
  auto c = relay::VarNode::make("c", tensor_type);
  auto y = relay::CallNode::make(add_op, {x, c}, tvm::Attrs(), {});
  auto func = relay::FunctionNode::make(relay::FreeVars(y), y, relay::Type(), {});
  auto A = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto B = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto C = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  auto pA = (float*)A.ToDLPack()->dl_tensor.data;
  auto pB = (float*)B.ToDLPack()->dl_tensor.data;
  auto pC = (float*)C.ToDLPack()->dl_tensor.data;

  for (int i = 0; i < 6; ++i) {
    pA[i] = i;
    pB[i] = i + 1;
    pC[i] = i + 2;
  }
  // get schedule
  auto reg = tvm::runtime::Registry::Get("relay.op._Register");
  auto s_i = tvm::runtime::Registry::Get("test.sch");
  if (!reg) {
    LOG(FATAL) << "no _Register";
  }
  if (!s_i) {
    LOG(FATAL) << "no _Register";
  }
  (*reg)("add", "FTVMSchedule", *s_i, 10);
  // build
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);
  Map<tvm::Integer, tvm::Target> targets;
  Target llvm_tgt = Target::Create("llvm");
  targets.Set(0, llvm_tgt);
  build_f(func, targets, llvm_tgt);
  std::string json = json_f();
  tvm::runtime::Module mod = mod_f();
  // run
  auto ctx = A->ctx;
  auto pfr = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
  tvm::runtime::Module run_mod = (*pfr)(json, mod, (int)ctx.device_type, (int)ctx.device_id);
  auto set_input_f = run_mod.GetFunction("set_input_zero_copy", false);
  auto run_f = run_mod.GetFunction("run", false);
  auto get_output_f = run_mod.GetFunction("get_output", false);
  set_input_f("a", &A.ToDLPack()->dl_tensor);
  set_input_f("b", &B.ToDLPack()->dl_tensor);
  set_input_f("c", &C.ToDLPack()->dl_tensor);
  run_f();
  tvm::runtime::NDArray Y = get_output_f(0);
  auto pY = (float*)Y.ToDLPack()->dl_tensor.data;
  for (int i = 0; i < 6; ++i) {
    CHECK_LT(fabs(pY[i] - (i + (i + 1) + (i + 2))), 1e-4);
  }
  // mutate the input a bit and run it again
  for (int i = 0; i < 6; ++i) {
    pB[i] = i + 3;
  }
  run_f();
  tvm::runtime::NDArray Y2 = get_output_f(0);
  auto pY2 = (float*)Y2.ToDLPack()->dl_tensor.data;
  for (int i = 0; i < 6; ++i) {
    CHECK_LT(fabs(pY2[i] - (i + (i + 3) + (i + 2))), 1e-4);
  }
  // attach a different input and run it again
  auto C2 = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto pC2 = (float*)C2.ToDLPack()->dl_tensor.data;
  for (int i = 0; i < 6; ++i) {
    pC2[i] = i + 4;
  }
  set_input_f("c", &C2.ToDLPack()->dl_tensor);
  run_f();
  tvm::runtime::NDArray Y3 = get_output_f(0);
  auto pY3 = (float*)Y3.ToDLPack()->dl_tensor.data;
  for (int i = 0; i < 6; ++i) {
    CHECK_LT(fabs(pY3[i] - (i + (i + 3) + (i + 4))), 1e-4);
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
