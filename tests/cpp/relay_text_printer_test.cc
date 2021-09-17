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
#include <tvm/ir/expr.h>
#include <tvm/ir/type_functor.h>
#include <tvm/node/functor.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
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
using namespace tvm::relay;

TEST(Relay, LargeGraphPrint) {
  auto foo = [] {
    auto add_op = relay::Op::Get("add");
    auto c_data = tvm::runtime::NDArray::Empty({1, 2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto c1 = relay::Constant(c_data);
    Call y1 = relay::Call(add_op, {c1, c1});
    for (int i = 0; i < 1e6; i++) {
      y1 = relay::Call(add_op, {c1, y1});
    }
    relay::Function func = relay::Function({}, y1, relay::Type(), {});
    std::string result = AsText(func);
    ASSERT_GT(0, result.size());
  };
  ASSERT_EXIT((foo(), exit(0)), ::testing::ExitedWithCode(0), ".*");
}
