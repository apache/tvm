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
#include <tvm/ir/memory_pools.h>
#include <tvm/ir/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/generic/injective.h>

using namespace tvm;
using namespace tvm::relay;

TVM_REGISTER_GLOBAL("runtime_test.strategy")
    .set_body_typed([](const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type,
                       const Target& target) {
      FTVMCompute fcompute = [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) -> Array<te::Tensor> {
        ICHECK_EQ(inputs.size(), 2U);
        return {topi::add(inputs[0], inputs[1])};
      };
      FTVMSchedule fschedule = [](const Attrs& attrs, const Array<te::Tensor>& outs,
                                  const Target& target) {
        With<Target> target_scope(target);
        return topi::generic::schedule_injective(target, outs);
      };

      auto n = make_object<OpStrategyNode>();
      auto strategy = tvm::relay::OpStrategy(std::move(n));
      strategy.AddImplementation(fcompute, fschedule, "runtime_test.strategy", 10);
      return strategy;
    });

TEST(Runtime, ZeroCopy) {
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
  auto Y = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});

  auto pA = static_cast<float*>(A->data);
  auto pB = static_cast<float*>(B->data);
  auto pC = static_cast<float*>(C->data);
  auto pY = static_cast<float*>(Y->data);

  for (int i = 0; i < 6; ++i) {
    pA[i] = i;
    pB[i] = i + 1;
    pC[i] = i + 2;
  }
  // get schedule
  auto reg = tvm::runtime::Registry::Get("ir.RegisterOpAttr");
  if (!reg) {
    LOG(FATAL) << "no _Register";
  }
  auto reset = tvm::runtime::Registry::Get("ir.OpResetAttr");
  if (!reset) {
    LOG(FATAL) << "Reset is not defined.";
  }
  auto fs = tvm::runtime::Registry::Get("runtime_test.strategy");
  if (!fs) {
    LOG(FATAL) << "No test_strategy registered.";
  }
  auto fgeneric = GenericFunc::Get("runtime_test.strategy_generic").set_default(*fs, true);
  (*reset)(add_op, "FTVMStrategy");
  (*reg)("add", "FTVMStrategy", fgeneric, 10);
  Array<Integer> dep;
  dep.push_back(0);
  (*reset)(add_op, "TShapeDataDependent");
  (*reg)("add", "TShapeDataDependent", dep, 10);
  // build
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  tvm::runtime::Module build_mod = (*pfb)();
  auto build_f = build_mod.GetFunction("build", false);
  auto json_f = build_mod.GetFunction("get_graph_json", false);
  auto mod_f = build_mod.GetFunction("get_module", false);
  Target llvm_tgt = Target("llvm");
  Array<Target> targets = {llvm_tgt};
  auto relay_mod = tvm::IRModule::FromExpr(func);
  ICHECK(relay_mod.defined()) << "Module must be defined";
  build_f(relay_mod, targets, llvm_tgt, Executor::Create("graph"), Runtime::Create("cpp"),
          WorkspaceMemoryPools(), ConstantMemoryPools(), "");
  // create graph executor
  std::string json = json_f();
  tvm::runtime::Module mod = mod_f();
  auto dev = A->device;
  auto pfr = tvm::runtime::Registry::Get("tvm.graph_executor.create");
  ICHECK(mod.defined()) << "Module must be defined";
  tvm::runtime::Module run_mod =
      (*pfr)(json, mod, static_cast<int>(dev.device_type), dev.device_id);
  // get function
  auto set_input_f = run_mod.GetFunction("set_input_zero_copy", false);
  auto set_output_f = run_mod.GetFunction("set_output_zero_copy", false);
  auto run_f = run_mod.GetFunction("run", false);
  // set input zero copy
  set_input_f("a", const_cast<DLTensor*>(A.operator->()));
  set_input_f("b", const_cast<DLTensor*>(B.operator->()));
  set_input_f("c", const_cast<DLTensor*>(C.operator->()));
  // set output zero copy
  set_output_f(0, const_cast<DLTensor*>(Y.operator->()));
  run_f();
  // check correctness
  for (int i = 0; i < 6; ++i) {
    ICHECK_LT(fabs(pY[i] - (i + (i + 1) + (i + 2))), 1e-4);
  }
  // mutate the input a bit and run it again
  for (int i = 0; i < 6; ++i) {
    pB[i] = i + 3;
  }
  run_f();
  // check correctness
  for (int i = 0; i < 6; ++i) {
    ICHECK_LT(fabs(pY[i] - (i + (i + 3) + (i + 2))), 1e-4);
  }
  // attach a different input and run it again
  auto C2 = tvm::runtime::NDArray::Empty({2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  auto pC2 = static_cast<float*>(C2->data);
  for (int i = 0; i < 6; ++i) {
    pC2[i] = i + 4;
  }
  set_input_f("c", const_cast<DLTensor*>(C2.operator->()));
  run_f();
  // check correctness
  for (int i = 0; i < 6; ++i) {
    ICHECK_LT(fabs(pY[i] - (i + (i + 3) + (i + 4))), 1e-4);
  }
}
