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
#include <tvm/driver/driver_api.h>
#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/generic/injective.h>

using namespace tvm;

TVM_REGISTER_GLOBAL("test.seq.strategy")
    .set_body_typed([](const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type,
                       const Target& target) {
      relay::FTVMCompute fcompute = [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                       const Type& out_type) -> Array<te::Tensor> {
        ICHECK_EQ(inputs.size(), 2U);
        return {topi::add(inputs[0], inputs[1])};
      };
      relay::FTVMSchedule fschedule = [](const Attrs& attrs, const Array<te::Tensor>& outs,
                                         const Target& target) {
        With<Target> target_scope(target);
        return topi::generic::schedule_injective(target, outs);
      };

      auto n = make_object<relay::OpStrategyNode>();
      auto strategy = relay::OpStrategy(std::move(n));
      strategy.AddImplementation(fcompute, fschedule, "test.strategy", 10);
      return strategy;
    });

TEST(Relay, Sequential) {
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

  auto reg = tvm::runtime::Registry::Get("ir.RegisterOpAttr");
  if (!reg) {
    LOG(FATAL) << "Register is not defined.";
  }
  auto reset = tvm::runtime::Registry::Get("ir.OpResetAttr");
  if (!reset) {
    LOG(FATAL) << "Reset is not defined.";
  }
  auto fs = tvm::runtime::Registry::Get("test.seq.strategy");
  if (!fs) {
    LOG(FATAL) << "Strategy is not defined.";
  }
  auto fgeneric = GenericFunc::Get("test.strategy_generic").set_default(*fs, true);
  (*reset)(add_op, "FTVMStrategy");
  (*reg)("add", "FTVMStrategy", fgeneric, 10);

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
    tvm::With<tvm::Target> tctx(tvm::Target("llvm"));
    mod = seq(mod);
  }

  ICHECK(mod.defined());
  auto entry_func = mod->GetGlobalVar("main");
  ICHECK(entry_func.defined());
  relay::Function f = Downcast<relay::Function>(mod->Lookup("main"));
  ICHECK(f.defined());

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
  ICHECK(tvm::StructuralEqual()(f, expected));
}

TEST(PassContextListConfigs, Basic) {
  Map<String, Map<String, String>> configs = relay::transform::PassContext::ListConfigs();
  ICHECK_EQ(configs.empty(), false);

  auto config = configs["relay.backend.use_auto_scheduler"];
  ICHECK_EQ(config["type"], "IntImm");
}
