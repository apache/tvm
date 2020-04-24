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
#include <tvm/te/operation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/op_attr_types.h>
#include <topi/broadcast.h>
#include <topi/generic/injective.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

using namespace tvm;
using namespace tvm::relay;

TVM_REGISTER_GLOBAL("test.strategy")
.set_body_typed([](const Attrs& attrs, const Array<te::Tensor>& inputs,
                   const Type& out_type, const Target& target) {
    FTVMCompute fcompute = [](const Attrs& attrs,
                              const Array<te::Tensor>& inputs,
                              const Type& out_type) -> Array<te::Tensor> {
        CHECK_EQ(inputs.size(), 2U);
        return {topi::add(inputs[0], inputs[1])};
    };
    FTVMSchedule fschedule = [](const Attrs& attrs,
                                const Array<te::Tensor>& outs,
                                const Target& target) {
        With<Target> target_scope(target);
        return topi::generic::schedule_injective(target, outs);
    };

    auto n = make_object<OpStrategyNode>();
    auto strategy = tvm::relay::OpStrategy(std::move(n));
    strategy.AddImplementation(fcompute, fschedule, "test.strategy", 10);
    return strategy;
});

TVM_REGISTER_GLOBAL("relay.backend.lower_call")
.set_body_typed([](const relay::Call& call, const Array<te::Tensor>& inputs,
                   const Target& target) {
    static auto fstrategy = Op::GetAttr<relay::FTVMStrategy>("FTVMStrategy");
    Op op = Downcast<Op>(call->op);
    auto out_type = call->checked_type();
    OpStrategy strategy = fstrategy[op](call->attrs, inputs, out_type, target);
    auto impl = strategy->specializations[0]->implementations[0];
    auto outs = impl.Compute(call->attrs, inputs, out_type);
    auto f = tvm::runtime::Registry::Get("relay.backend._make_LoweredOutput");
    if (!f) {
      LOG(FATAL) << "relay.backend._make_LoweredOutput is not registered";
    }
    return (*f)(outs, impl);
});

TEST(Relay, BuildModule) {
  auto tensor_type = relay::TensorType({2, 3}, DataType::Float(32));
  auto a = relay::Var("a", tensor_type);
  auto b = relay::Var("b", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto x = relay::Call(add_op, {a, b}, tvm::Attrs(), {});
  auto c = relay::Var("c", tensor_type);
  auto y = relay::Call(add_op, {x, c}, tvm::Attrs(), {});
  auto func = relay::Function(relay::FreeVars(y), y, relay::Type(), {});
  auto A = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto B = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto C = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  auto pA = (float*)A->data;
  auto pB = (float*)B->data;
  auto pC = (float*)C->data;

  for (int i = 0; i < 6; ++i) {
    pA[i] = i;
    pB[i] = i + 1;
    pC[i] = i + 2;
  }
  // get schedule
  auto reg = tvm::runtime::Registry::Get("relay.op._Register");
  if (!reg) {
    LOG(FATAL) << "no _Register";
  }
  auto fs = tvm::runtime::Registry::Get("test.strategy");
  if (!fs) {
    LOG(FATAL) << "No test_strategy registered.";
  }
  auto fgeneric = GenericFunc::Get("test.strategy_generic").set_default(*fs);
  (*reg)("add", "FTVMStrategy", fgeneric, 10);
  // build
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);
  Map<tvm::Integer, tvm::Target> targets;
  Target llvm_tgt = Target::Create("llvm");
  targets.Set(0, llvm_tgt);
  auto relay_mod = tvm::IRModule::FromExpr(func);
  build_f(relay_mod, targets, llvm_tgt);
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
  auto pY = (float*)Y->data;
  for (int i = 0; i < 6; ++i) {
    CHECK_LT(fabs(pY[i] - (i + (i + 1) + (i + 2))), 1e-4);
  }
  // mutate the input a bit and run it again
  for (int i = 0; i < 6; ++i) {
    pB[i] = i + 3;
  }
  run_f();
  tvm::runtime::NDArray Y2 = get_output_f(0);
  auto pY2 = (float*)Y2->data;
  for (int i = 0; i < 6; ++i) {
    CHECK_LT(fabs(pY2[i] - (i + (i + 3) + (i + 2))), 1e-4);
  }
  // attach a different input and run it again
  auto C2 = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto pC2 = (float*)C2->data;
  for (int i = 0; i < 6; ++i) {
    pC2[i] = i + 4;
  }
  set_input_f("c", &C2.ToDLPack()->dl_tensor);
  run_f();
  tvm::runtime::NDArray Y3 = get_output_f(0);
  auto pY3 = (float*)Y3->data;
  for (int i = 0; i < 6; ++i) {
    CHECK_LT(fabs(pY3[i] - (i + (i + 3) + (i + 4))), 1e-4);
  }
}

TEST(Relay, GetExprRefCount) {
  auto tensor_type = relay::TensorType({2, 3}, DataType::Float(32));
  auto a = relay::Var("a", tensor_type);
  auto add_op = relay::Op::Get("add");
  auto relu_op = relay::Op::Get("nn.relu");
  auto x = relay::Call(relu_op, {a}, tvm::Attrs(), {});
  auto y = relay::Call(relu_op, {x}, tvm::Attrs(), {});
  auto z = relay::Call(add_op, {y, x}, tvm::Attrs(), {});
  auto ref_count = GetExprRefCount(z);
  CHECK(ref_count[a.get()] == 1);
  CHECK(ref_count[relu_op.get()] == 2);
  CHECK(ref_count[add_op.get()] == 1);
  CHECK(ref_count[x.get()] == 2);
  CHECK(ref_count[y.get()] == 1);
  CHECK(ref_count[z.get()] == 1);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
