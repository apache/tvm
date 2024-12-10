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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/type.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>
#include <tvm/te/operation.h>

TEST(CCodegen, MainFunctionOrder) {
  using namespace tvm;
  using namespace tvm::te;

  std::string tvm_module_main = std::string(runtime::symbol::tvm_module_main);

  tvm::Target target_c = tvm::Target("c -keys=cpu");

  const int n = 4;
  Array<PrimExpr> shape{n};

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");

  auto elemwise_add = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "elemwise_add");

  auto fcreate = [=]() {
    With<Target> llvm_scope(target_c);
    return create_schedule({elemwise_add->op});
  };

  auto args = Array<Tensor>({A, B, elemwise_add});

  std::unordered_map<Tensor, Buffer> binds;
  auto lowered = LowerSchedule(fcreate(), args, "elemwise_add", binds, GlobalVarSupply());
  Map<tvm::Target, IRModule> inputs = {{target_c, lowered}};
  runtime::Module module = build(inputs, Target());
  Array<String> functions = module->GetFunction("get_func_names", false)();

  ICHECK(functions.back().compare(tvm_module_main) == 0);
}

auto BuildLowered(std::string op_name, tvm::Target target) {
  using namespace tvm;
  using namespace tvm::te;

  // The shape of input tensors.
  const int n = 4;
  Array<PrimExpr> shape{n};

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");

  auto op = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, op_name);

  auto fcreate_s = [=]() {
    With<Target> llvm_scope(target);
    return create_schedule({op->op});
  };

  auto args = Array<Tensor>({A, B, op});
  std::unordered_map<Tensor, Buffer> binds;
  auto lowered_s = LowerSchedule(fcreate_s(), args, op_name, binds, GlobalVarSupply());
  return lowered_s;
}

bool IsSorted(tvm::Map<tvm::Target, tvm::IRModule> inputs) {
  std::vector<std::string> schedule_names;
  for (auto const& module : inputs) {
    for (auto const& func : module.second->functions) {
      schedule_names.push_back(func.first->name_hint);
    }
  }
  return std::is_sorted(schedule_names.begin(), schedule_names.end());
}

TEST(CCodegen, FunctionOrder) {
  using testing::_;
  using testing::ElementsAre;
  using testing::StrEq;
  using namespace tvm;
  using namespace tvm::te;

  Target target = Target("c -keys=cpu");

  // add schedules in reverse order
  Map<tvm::Target, IRModule> inputs;
  inputs.Set(Target("c -keys=cpu"), BuildLowered("op_2", target));
  inputs.Set(Target("c -keys=cpu"), BuildLowered("op_1", target));

  for (uint32_t counter = 99; IsSorted(inputs) && counter > 0; counter--) {
    std::string op_name = "op_" + std::to_string(counter);
    inputs.Set(Target("c -keys=cpu"), BuildLowered(op_name, target));
  }

  EXPECT_FALSE(IsSorted(inputs));

  auto module = build(inputs, Target());
  Array<String> func_array = module->GetFunction("get_func_names", false)();
  std::vector<std::string> functions{func_array.begin(), func_array.end()};
  // The entry point is handled separately from the other functions.
  functions.erase(std::remove_if(functions.begin(), functions.end(),
                                 [](const std::string& name) {
                                   return name == tvm::runtime::symbol::tvm_module_main;
                                 }),
                  functions.end());
  EXPECT_TRUE(std::is_sorted(functions.begin(), functions.end()));
}
