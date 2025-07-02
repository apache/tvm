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
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/rvalue_ref.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(RValueRef, Basic) {
  auto append =
      Function::FromTyped([](RValueRef<Array<int>> ref, int val, bool is_unique) -> Array<int> {
        Array<int> arr = *std::move(ref);
        EXPECT_EQ(arr.unique(), is_unique);
        arr.push_back(val);
        return arr;
      });
  auto a = append(RValueRef(Array<int>({1, 2})), 3, true).cast<Array<int>>();
  EXPECT_EQ(a.size(), 3);
  a = append(RValueRef(std::move(a)), 4, true).cast<Array<int>>();
  EXPECT_EQ(a.size(), 4);
  // pass in lvalue instead, the append still will succeed but array will not be unique
  a = append(a, 5, false).cast<Array<int>>();
  EXPECT_EQ(a.size(), 5);
}

TEST(RValueRef, ParamChecking) {
  // try decution
  Function fadd1 = Function::FromTyped([](TInt a) -> int64_t { return a->value + 1; });

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd1(RValueRef(TInt(1)));
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(error.message(),
                    "Mismatched type on argument #0 when calling: `(0: test.Int) -> int`. "
                    "Expected `test.Int` but got `ObjectRValueRef`");
          throw;
        }
      },
      ::tvm::ffi::Error);

  Function fadd2 = Function::FromTyped([](RValueRef<Array<int>> a) -> int {
    Array<int> arr = *std::move(a);
    return arr[0] + 1;
  });

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd2(RValueRef(Array<Any>({1, 2.2})));
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(
              error.message(),
              "Mismatched type on argument #0 when calling: `(0: RValueRef<Array<int>>) -> int`. "
              "Expected `RValueRef<Array<int>>` but got `RValueRef<Array[index 1: float]>`");
          throw;
        }
      },
      ::tvm::ffi::Error);
  // triggered a rvalue based conversion
  Function func3 = Function::FromTyped([](RValueRef<TPrimExpr> a) -> String {
    TPrimExpr expr = *std::move(a);
    return expr->dtype;
  });
  EXPECT_EQ(func3(RValueRef(String("int32"))).cast<String>(), "int32");
  // triggered a lvalue based conversion
  EXPECT_EQ(func3(String("int32")).cast<String>(), "int32");
}
}  // namespace
