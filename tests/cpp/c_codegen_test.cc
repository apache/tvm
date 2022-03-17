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

  tvm::Target target_c = tvm::Target("c -keys=cpu -link-params=0");

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
  auto lowered = LowerSchedule(fcreate(), args, "elemwise_add", binds);
  Map<tvm::Target, IRModule> inputs = {{target_c, lowered}};
  runtime::Module module = build(inputs, Target());
  Array<String> functions = module->GetFunction("get_func_names", false)();

  ICHECK(functions.back().compare(tvm_module_main) == 0);
}

TEST(CCodegen, FunctionOrder) {
  using testing::_;
  using testing::ElementsAre;
  using testing::StrEq;
  using namespace tvm;
  using namespace tvm::te;

  auto target = Target("c -keys=cpu -link-params=0");

  // The shape of input tensors.
  const int n = 4;
  Array<PrimExpr> shape{n};

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");

  auto op_1 = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "op_1");

  auto op_2 = compute(
      A->shape, [&A, &B](PrimExpr i) { return A[i] - B[i]; }, "op_2");

  auto fcreate_s1 = [=]() {
    With<Target> llvm_scope(target);
    return create_schedule({op_1->op});
  };

  auto fcreate_s2 = [=]() {
    With<Target> llvm_scope(target);
    return create_schedule({op_2->op});
  };

  auto args1 = Array<Tensor>({A, B, op_1});
  auto args2 = Array<Tensor>({A, B, op_2});

  std::unordered_map<Tensor, Buffer> binds;
  auto lowered_s1 = LowerSchedule(fcreate_s1(), args1, "op_1", binds);
  auto lowered_s2 = LowerSchedule(fcreate_s2(), args2, "op_2", binds);

  // add schedules in reverse order
  Map<tvm::Target, IRModule> inputs = {{target, lowered_s2}, {target, lowered_s1}};
  std::vector<std::string> schedule_names;
  for (auto const& module : inputs) {
    for (auto const& func : module.second->functions) {
      schedule_names.push_back(func.first->name_hint);
    }
  }
  EXPECT_THAT(schedule_names, ElementsAre(StrEq("op_2"), StrEq("op_1")));

  auto module = build(inputs, Target());
  Array<String> func_array = module->GetFunction("get_func_names", false)();
  std::vector<std::string> functions{func_array.begin(), func_array.end()};
  EXPECT_THAT(functions, ElementsAre(StrEq("op_1"), _, StrEq("op_2"), _));
}
