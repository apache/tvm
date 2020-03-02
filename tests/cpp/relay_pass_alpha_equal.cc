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

using namespace tvm;

class TestAlphaEquals {
  runtime::PackedFunc *_packed_func;
 public:
  TestAlphaEquals(const char* func_name) {
    _packed_func = new runtime::PackedFunc();
    TVMFuncGetGlobal(func_name, reinterpret_cast<TVMFunctionHandle*>(&_packed_func));
  }

  void UpdatePackedFunc(const char* func_name) {
    TVMFuncGetGlobal(func_name, reinterpret_cast<TVMFunctionHandle*>(&_packed_func));
  }

  bool operator()(ObjectRef input_1, ObjectRef input_2) {
    TVMRetValue rv;
    std::vector<TVMValue> values(2);
    std::vector<int> codes(2);
    runtime::TVMArgsSetter setter(values.data(), codes.data());
    setter(0, input_1);
    setter(1, input_2);
    _packed_func->CallPacked(TVMArgs(values.data(), codes.data(), 2), &rv);
    return bool(rv);
  };

};

TEST(Relay, AlphaTestEmptyTypeNodes) {
  auto x = TypeVar("x", kTypeData);
  auto y = TypeVar();
  EXPECT_FALSE(relay::AlphaEqual(x, y));

  TestAlphaEquals test_equals("relay._make._alpha_equal");
  EXPECT_FALSE(test_equals(x, y));
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
