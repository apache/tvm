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

#include <memory>

using namespace tvm;
using namespace tvm::relay;

TEST(Relay, OutOfStack_add) {
  auto foo = [] {
    auto add_op = relay::Op::Get("add");
    auto c_data = tvm::runtime::NDArray::Empty({1, 2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto c1 = relay::Constant(c_data);
    Call y1 = relay::Call(add_op, {c1, c1});
    for (int i = 0; i < 1e6; i++) {
      y1 = relay::Call(add_op, {c1, y1});
    }
    relay::Function func = relay::Function({}, y1, relay::Type(), {});
  };
  ASSERT_EXIT((foo(), exit(0)), ::testing::ExitedWithCode(0), ".*");
}

TEST(Relay, OutOfStack_cast) {
  auto foo = [] {
    auto cast_op = relay::Op::Get("cast");
    auto c_data = tvm::runtime::NDArray::Empty({1, 2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto c1 = relay::Constant(c_data);
    Call y1 = relay::Call(cast_op, {c1});
    for (int i = 0; i < 1e6; i++) {
      y1 = relay::Call(cast_op, {y1});
    }
    relay::Function func = relay::Function({}, y1, relay::Type(), {});
  };
  ASSERT_EXIT((foo(), exit(0)), ::testing::ExitedWithCode(0), ".*");
}

TEST(Relay, OutOfStack_packed_func) {
  constexpr int len = 1e6;
  auto foo = [] {
    auto x = relay::Var("x", relay::TensorType({3, 2}, DataType::Float(32)));
    auto one = relay::Constant(tvm::runtime::NDArray::Empty({1}, {kDLFloat, 32, 1}, {kDLCPU, 0}));
    auto add_func = tvm::runtime::Registry::Get("relay.op._make.add");
    auto y = (*add_func)(x, one);
    for (int i = 0; i < len; ++i) {
      y = (*add_func)(y, one);
    }

    // check if still reachable
    int k = 0;
    Expr e = y;
    while (e.defined() && e.as<CallNode>() != nullptr) {
      e = e.as<CallNode>()->args[0];
      ++k;
    }
    ASSERT_EQ(len + 1, k);
  };
  ASSERT_EXIT((foo(), exit(0)), ::testing::ExitedWithCode(0), ".*");
}

TEST(Relay, CallNodeSharedArgs) {
  auto x = relay::Var("x", relay::TensorType({3, 2}, DataType::Float(32)));
  auto one = relay::Constant(tvm::runtime::NDArray::Empty({1}, {kDLFloat, 32, 1}, {kDLCPU, 0}));
  auto relu_op = relay::Op::Get("nn.relu");
  Call y = relay::Call(relu_op, {x}, Attrs(), {});
  y = relay::Call(relu_op, {y}, Attrs(), {});
  ASSERT_EQ(1, y.get()->args[0].as<CallNode>()->args.size());
  y = relay::Call(y.get()->op, y.get()->args, y.get()->attrs, y.get()->type_args);
  ASSERT_EQ(1, y.get()->args[0].as<CallNode>()->args.size());
}

TEST(Relay, TupleSharedFields) {
  auto x = relay::Var("x", relay::TensorType({3, 2}, DataType::Float(32)));
  auto one = relay::Constant(tvm::runtime::NDArray::Empty({1}, {kDLFloat, 32, 1}, {kDLCPU, 0}));
  auto relu_op = relay::Op::Get("nn.relu");
  Expr y = relay::Call(relu_op, {x}, Attrs(), {});
  y = relay::Call(relu_op, {y}, Attrs(), {});
  {
    Expr y1 = relay::Tuple(y.as<CallNode>()->args);
    Expr y2 = relay::Tuple(y.as<CallNode>()->args);

    y1 = relay::Call(relu_op, {y1});
    y2 = relay::Call(relu_op, {y2});
    y = y1;
  }
  ASSERT_EQ(1, y.as<CallNode>()->args[0].as<TupleNode>()->fields[0].as<CallNode>()->args.size());
}

TEST(Relay, TupleiGetItemSharedTuple) {
  auto x = relay::Var("x", relay::TensorType({3, 2}, DataType::Float(32)));
  auto one = relay::Constant(tvm::runtime::NDArray::Empty({1}, {kDLFloat, 32, 1}, {kDLCPU, 0}));
  auto relu_op = relay::Op::Get("nn.relu");
  Expr y = relay::Call(relu_op, {x}, Attrs(), {});
  y = relay::Tuple({y});
  {
    Expr y1 = relay::TupleGetItem(y, 0);
    Expr y2 = relay::TupleGetItem(y, 0);

    y1 = relay::Call(relu_op, {y1});
    y2 = relay::Call(relu_op, {y2});
    y = y1;
  }
  ASSERT_EQ(1, y.as<CallNode>()
                   ->args[0]
                   .as<TupleGetItemNode>()
                   ->tuple.as<TupleNode>()
                   ->fields[0]
                   .as<CallNode>()
                   ->args.size());
}

TEST(Relay, OutOfStackLet) {
  auto foo = [] {
    auto add_op = relay::Op::Get("add");
    auto p = relay::Var("p", relay::TensorType({3, 2}, DataType::Float(32)));
    int size = 1e6 - 1;
    std::vector<relay::Var> vars;
    for (int i = 0; i < size; ++i) {
      vars.emplace_back("x_" + std::to_string(i), relay::TensorType({3, 2}, DataType::Float(32)));
    }
    Expr body = vars[size - 1];
    for (int i = size - 1; i >= 0; --i) {
      Var v = i == 0 ? p : vars[i - 1];
      body = relay::Let(vars[i], relay::Call(add_op, {v, v}), body);
    }
    relay::Function func = relay::Function({p}, body, relay::Type(), {});
  };
  ASSERT_EXIT((foo(), exit(0)), ::testing::ExitedWithCode(0), ".*");
}
