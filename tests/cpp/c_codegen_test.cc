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

TEST(CCodegen, FunctionOrder) {
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
