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
#include <topi/generic/injective.h>
#include <tvm/build_module.h>
#include <tvm/packed_func_ext.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/tvm.h>

TVM_REGISTER_GLOBAL("schedule")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      *rv = topi::generic::schedule_injective(args[0], args[1]);
    });

TEST(Relay, Sequential) {
  using namespace tvm;
  auto tensor_type = relay::TensorTypeNode::make({1, 2, 3}, ::tvm::Float(32));
  auto c_data =
      tvm::runtime::NDArray::Empty({1, 2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  // Create a function for optimization.
  auto c = relay::ConstantNode::make(c_data);
  auto a = relay::VarNode::make("a", tensor_type);
  auto x = relay::VarNode::make("x", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto y = relay::CallNode::make(add_op, {c, c});
  y = relay::CallNode::make(add_op, {x, y});
  auto z = relay::CallNode::make(add_op, {y, c});
  auto z1 = relay::CallNode::make(add_op, {y, c});
  auto z2 = relay::CallNode::make(add_op, {z, z1});
  // Let expression and varaible a should be dead-code eliminated.
  auto z3 = relay::LetNode::make(a, c, z2);
  relay::Function func =
      relay::FunctionNode::make(relay::FreeVars(z3), z3, relay::Type(), {});

  // Get schedule
  auto reg = tvm::runtime::Registry::Get("relay.op._Register");
  auto sch = tvm::runtime::Registry::Get("schedule");
  if (!reg || !sch) {
    LOG(FATAL) << "Register/schedule is not defined.";
  }

  (*reg)("add", "FTVMSchedule", *sch, 10);

  // Run sequential passes.
  tvm::Array<relay::transform::Pass> pass_seqs;
  pass_seqs.push_back(relay::transform::InferType());
  pass_seqs.push_back(relay::transform::DeadCodeElimination());
  pass_seqs.push_back(relay::transform::EliminateCommonSubexpr());
  pass_seqs.push_back(relay::transform::AlterOpLayout());

  relay::GlobalVar var = relay::GlobalVarNode::make("main");
  tvm::Map<relay::GlobalVar, relay::Function> m;
  m.Set(var, func);
  auto mod = relay::ModuleNode::make(m, {});
  relay::transform::Pass seq = relay::transform::Sequential(pass_seqs);
  {
    tvm::With<relay::transform::PassContext> pass_ctx(
        relay::transform::PassContext(3, 1, {}, {}));
    tvm::With<tvm::Target> tctx(tvm::Target::Create("llvm"));
    mod = seq->operator()(mod);
  }

  CHECK(mod.defined());

  relay::Function f;
  for (const auto& kv : mod->functions) {
    f = kv.second;
  }
  CHECK(f.defined());

  // Expected function
  auto c1 = relay::ConstantNode::make(c_data);
  auto x1 = relay::VarNode::make("x", tensor_type);
  auto y1 = relay::CallNode::make(add_op, {c1, c1});
  y1 = relay::CallNode::make(add_op, {x1, y1});
  auto zz = relay::CallNode::make(add_op, {y1, c1});
  zz = relay::CallNode::make(add_op, {zz, zz});
  relay::Function expected_func =
      relay::FunctionNode::make(relay::FreeVars(zz), zz, relay::Type(), {});

  // Infer type for the expected function.
  auto expected = relay::InferType(expected_func, relay::Module(nullptr));
  CHECK(relay::AlphaEqual(f, expected));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
