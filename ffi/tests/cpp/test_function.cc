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
#include <tvm/ffi/memory.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Func, FromPacked) {
  Function fadd1 = Function::FromPacked([](int32_t num_args, const AnyView* args, Any* rv) {
    EXPECT_EQ(num_args, 1);
    int32_t a = args[0];
    *rv = a + 1;
  });
  int b = fadd1(1);
  EXPECT_EQ(b, 2);

  Function fadd2 = Function::FromPacked([](int32_t num_args, const AnyView* args, Any* rv) {
    EXPECT_EQ(num_args, 1);
    TInt a = args[0];
    EXPECT_EQ(a.use_count(), 2);
    *rv = a->value + 1;
  });
  EXPECT_EQ(fadd2(TInt(12)).operator int(), 13);
}

TEST(Func, FromUnpacked) {
  // try decution
  Function fadd1 = Function::FromUnpacked([](const int32_t& a) -> int { return a + 1; });
  int b = fadd1(1);
  EXPECT_EQ(b, 2);

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd1(1.1);
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          EXPECT_STREQ(error->message.c_str(),
                       "Mismatched type on argument #0 when calling: `(0: int) -> int`. "
                       "Expected `int` but got `float`");
          throw;
        }
      },
      ::tvm::ffi::Error);

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd1();
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          EXPECT_STREQ(error->message.c_str(),
                       "Mismatched number of arguments when calling: `(0: int) -> int`. "
                       "Expected 1 but got 0 arguments");
          throw;
        }
      },
      ::tvm::ffi::Error);

  // try decution
  Function fpass_and_return = Function::FromUnpacked(
      [](TInt x, int value, AnyView z) -> Function {
        EXPECT_EQ(x.use_count(), 2);
        EXPECT_EQ(x->value, value);
        if (auto opt = z.TryAs<int>()) {
          EXPECT_EQ(value, *opt);
        }
        return Function::FromUnpacked([value](int x) -> int { return x + value; });
      },
      "fpass_and_return");
  TInt a(11);
  Function fret = fpass_and_return(std::move(a), 11, 11);
  EXPECT_EQ(fret(12).operator int(), 23);

  EXPECT_THROW(
      {
        try {
          fpass_and_return();
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          EXPECT_STREQ(error->message.c_str(),
                       "Mismatched number of arguments when calling: "
                       "`fpass_and_return(0: test.Int, 1: int, 2: AnyView) -> object.Function`. "
                       "Expected 3 but got 0 arguments");
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(Func, Global) {
  Function::SetGlobal("testing.add1",
                      Function::FromUnpacked([](const int32_t& a) -> int { return a + 1; }));
  Function fadd1 = Function::GetGlobal("testing.add1");
  int b = fadd1(1);
  EXPECT_EQ(b, 2);
  Function fnot_exist = Function::GetGlobal("testing.not_existing_func");
  EXPECT_TRUE(fnot_exist == nullptr);

  Array<String> names = Function::GetGlobal("tvm_ffi.GlobalFunctionListNames")();

  EXPECT_TRUE(std::find(names.begin(), names.end(), "testing.add1") != names.end());
}

}  // namespace
