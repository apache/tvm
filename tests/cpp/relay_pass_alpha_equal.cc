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
#include <tvm/te/operation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>

TEST(Relay, AlphaTestEmptyTypeNodes) {
  using namespace tvm;
  auto x = TypeVar("x", kTypeData);
  auto y = TypeVar();
  EXPECT_FALSE(relay::AlphaEqual(x, y));

  runtime::PackedFunc *packed_func = new tvm::runtime::PackedFunc();
  TVMRetValue rv;
  (void)TVMFuncGetGlobal("relay._make._alpha_equal", (void**)&packed_func);
  std::vector<TVMValue> values(2);
  std::vector<int> codes(2);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, x);
  setter(1, y);
  packed_func->CallPacked(TVMArgs(values.data(), codes.data(), 2), &rv);
  EXPECT_FALSE(bool(rv));
}

TEST(Relay, AlphaTestSameTypeNodes) {
  using namespace tvm;
  auto x = TypeVar("x", kTypeData);
  EXPECT_TRUE(relay::AlphaEqual(x, x));

  runtime::PackedFunc *packed_func = new tvm::runtime::PackedFunc();
  TVMRetValue rv;
  (void)TVMFuncGetGlobal("relay._make._alpha_equal", (void**)&packed_func);
  std::vector<TVMValue> values(2);
  std::vector<int> codes(2);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, x);
  setter(1, x);
  packed_func->CallPacked(TVMArgs(values.data(), codes.data(), 2), &rv);
  EXPECT_TRUE(bool(rv));
}

TEST(Relay, AlphaTestIncompatibleTypeNodes) {
  using namespace tvm;
  auto x = TypeVar("x", kTypeData);
  auto y = relay::VarNode::make("y", relay::Type());
  runtime::PackedFunc *packed_func = new tvm::runtime::PackedFunc();
  TVMRetValue rv;
  (void)TVMFuncGetGlobal("relay._make._alpha_equal", (void**)&packed_func);
  std::vector<TVMValue> values(2);
  std::vector<int> codes(2);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, x);
  setter(1, y);
  packed_func->CallPacked(TVMArgs(values.data(), codes.data(), 2), &rv);
  EXPECT_FALSE(bool(rv));
  
  setter(0, y);
  setter(1, x);
  packed_func->CallPacked(TVMArgs(values.data(), codes.data(), 2), &rv);
  EXPECT_FALSE(bool(rv));
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
