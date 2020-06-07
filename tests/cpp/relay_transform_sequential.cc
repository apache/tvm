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
#include <tvm/driver/driver_api.h>
#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

TVM_REGISTER_GLOBAL("schedule").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  *rv = topi::generic::schedule_injective(args[0], args[1]);
});

TEST(Relay, Sequential) {
  using namespace tvm;
  auto tensor_type = relay::TensorType({1, 2, 3}, DataType::Float(32));
  auto c_data = tvm::runtime::NDArray::Empty({1, 2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  // Create a function for optimization.
  auto c = relay::Constant(c_data);
  auto a = relay::Var("a", tensor_type);
  auto x = relay::Var("x", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto y = relay::Call(add_op, {c, c});
  y = relay::Call(add_op, {x, y});
  auto z = relay::Call(add_op, {y, c});
  auto z1 = relay::Call(add_op, {y, c});
  auto z2 = relay::Call(add_op, {z, z1});
  // Let expression and varaible a should be dead-code eliminated.
  auto z3 = relay::Let(a, c, z2);
  relay::Function func = relay::Function(relay::FreeVars(z3), z3, relay::Type(), {});

  // Get schedule
  auto reg = tvm::runtime::Registry::Get("relay.op._Register");
  auto sch = tvm::runtime::Registry::Get("schedule");
  if (!reg || !sch) {
    LOG(FATAL) << "Register/schedule is not defined.";
  }

  (*reg)("add", "FTVMSchedule", *sch, 10);

  // Run sequential passes.
  tvm::Array<relay::transform::Pass> pass_seqs{
      relay::transform::InferType(), relay::transform::DeadCodeElimination(),
      relay::transform::EliminateCommonSubexpr(), relay::transform::AlterOpLayout()};
  relay::transform::Pass seq = relay::transform::Sequential(pass_seqs);
  auto mod = IRModule::FromExpr(func);
  auto pass_ctx = relay::transform::PassContext::Create();
  pass_ctx->opt_level = 3;
  pass_ctx->config.Set("relay.fallback_device_type", Integer(1));
  {
    tvm::With<relay::transform::PassContext> ctx_scope(pass_ctx);
    tvm::With<tvm::Target> tctx(tvm::Target::Create("llvm"));
    mod = seq(mod);
  }

  CHECK(mod.defined());
  auto entry_func = mod->GetGlobalVar("main");
  CHECK(entry_func.defined());
  relay::Function f = Downcast<relay::Function>(mod->Lookup("main"));
  CHECK(f.defined());

  // Expected function
  auto c1 = relay::Constant(c_data);
  auto x1 = relay::Var("x", tensor_type);
  auto y1 = relay::Call(add_op, {c1, c1});
  y1 = relay::Call(add_op, {x1, y1});
  auto zz = relay::Call(add_op, {y1, c1});
  zz = relay::Call(add_op, {zz, zz});
  relay::Function expected_func = relay::Function(relay::FreeVars(zz), zz, relay::Type(), {});

  // Infer type for the expected function.
  auto mod1 = IRModule::FromExpr(expected_func);
  mod1 = relay::transform::InferType()(mod1);
  auto expected = mod1->Lookup("main");
  CHECK(tvm::StructuralEqual()(f, expected));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
