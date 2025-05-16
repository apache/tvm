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
  Function fadd1 = Function::FromPacked([](const AnyView* args, int32_t num_args, Any* rv) {
    EXPECT_EQ(num_args, 1);
    int32_t a = args[0].cast<int32_t>();
    *rv = a + 1;
  });
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);

  Function fadd2 = Function::FromPacked([](const AnyView* args, int32_t num_args, Any* rv) {
    EXPECT_EQ(num_args, 1);
    auto a = args[0].cast<TInt>();
    EXPECT_EQ(a.use_count(), 2);
    *rv = a->value + 1;
  });
  EXPECT_EQ(fadd2(TInt(12)).cast<int>(), 13);
}

TEST(Func, PackedArgs) {
  Function fadd1 = Function::FromPacked([](PackedArgs args, Any* rv) {
    EXPECT_EQ(args.size(), 1);
    int32_t a = args[0].cast<int>();
    *rv = a + 1;
  });
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);

  Function fadd2 = Function::FromPacked([](PackedArgs args, Any* rv) {
    EXPECT_EQ(args.size(), 1);
    TInt a = args[0].cast<TInt>();
    EXPECT_EQ(a.use_count(), 2);
    *rv = a->value + 1;
  });
  EXPECT_EQ(fadd2(TInt(12)).cast<int>(), 13);

  TInt v(12);
  AnyView data[3];
  PackedArgs::Fill(data, 3, 1, v);
  EXPECT_EQ(data[0].cast<int>(), 3);
  EXPECT_EQ(data[1].cast<int>(), 1);
  EXPECT_EQ(data[2].cast<TInt>()->value, 12);
}

TEST(Func, FromTyped) {
  // try decution
  Function fadd1 = Function::FromTyped([](const int32_t& a) -> int { return a + 1; });
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd1(1.1);
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(error.message(),
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
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(error.message(),
                    "Mismatched number of arguments when calling: `(0: int) -> int`. "
                    "Expected 1 but got 0 arguments");
          throw;
        }
      },
      ::tvm::ffi::Error);

  // try decution
  Function fpass_and_return = Function::FromTyped(
      [](TInt x, int value, AnyView z) -> Function {
        EXPECT_EQ(x.use_count(), 2);
        EXPECT_EQ(x->value, value);
        if (auto opt = z.as<int>()) {
          EXPECT_EQ(value, *opt);
        }
        return Function::FromTyped([value](int x) -> int { return x + value; });
      },
      "fpass_and_return");
  TInt a(11);
  auto fret = fpass_and_return(std::move(a), 11, 11).cast<Function>();
  EXPECT_EQ(fret(12).cast<int>(), 23);

  EXPECT_THROW(
      {
        try {
          fpass_and_return();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(error.message(),
                    "Mismatched number of arguments when calling: "
                    "`fpass_and_return(0: test.Int, 1: int, 2: AnyView) -> object.Function`. "
                    "Expected 3 but got 0 arguments");
          throw;
        }
      },
      ::tvm::ffi::Error);

  Function fconcact =
      Function::FromTyped([](const String& a, const String& b) -> String { return a + b; });
  EXPECT_EQ(fconcact("abc", "def").cast<String>(), "abcdef");
}

TEST(Func, PassReturnAny) {
  Function fadd_one = Function::FromTyped([](Any a) -> Any { return a.cast<int>() + 1; });
  EXPECT_EQ(fadd_one(1).cast<int>(), 2);
}

TEST(Func, Global) {
  Function::SetGlobal("testing.add1",
                      Function::FromTyped([](const int32_t& a) -> int { return a + 1; }));
  auto fadd1 = Function::GetGlobalRequired("testing.add1");
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);
  auto fnot_exist = Function::GetGlobal("testing.not_existing_func");
  EXPECT_TRUE(!fnot_exist);

  auto fname_functor =
      Function::GetGlobal("ffi.FunctionListGlobalNamesFunctor").value()().cast<Function>();
  Array<String> names;
  int len = fname_functor(-1).cast<int>();
  for (int i = 0; i < len; ++i) {
    names.push_back(fname_functor(i).cast<String>());
  }
  EXPECT_TRUE(std::find(names.begin(), names.end(), "testing.add1") != names.end());
}

TEST(Func, TypedFunction) {
  TypedFunction<int(int)> fadd1 = [](int a) -> int { return a + 1; };
  EXPECT_EQ(fadd1(1), 2);

  TypedFunction<int(int)> fadd2([](int a) -> int { return a + 2; });
  EXPECT_EQ(fadd2(1), 3);
  EXPECT_EQ(fadd2.packed()(1).cast<int>(), 3);

  TypedFunction<void(int)> fcheck_int;
  EXPECT_TRUE(fcheck_int == nullptr);
  fcheck_int = [](int a) -> void { EXPECT_EQ(a, 1); };
  fcheck_int(1);
}

TEST(Func, TypedFunctionAsAny) {
  TypedFunction<int(int)> fadd1 = [](int a) -> int { return a + 1; };
  Any fany(std::move(fadd1));
  EXPECT_TRUE(fadd1 == nullptr);
  auto fadd1_dup = fany.cast<TypedFunction<int(int)>>();
  EXPECT_EQ(fadd1_dup(1), 2);
}

TEST(Func, TypedFunctionAsAnyView) {
  TypedFunction<int(int)> fadd2 = [](int a) -> int { return a + 2; };
  AnyView fview(fadd2);
  auto fadd2_dup = fview.cast<TypedFunction<int(int)>>();
  EXPECT_EQ(fadd2_dup(1), 3);
}

TEST(Func, ObjectRefWithFallbackTraits) {
  // test cases to test automatic type conversion via ObjectRefWithFallbackTraits
  // through TPrimExpr
  Function freturn_primexpr = Function::FromTyped([](TPrimExpr a) -> TPrimExpr { return a; });

  auto result_int = freturn_primexpr(1).cast<TPrimExpr>();
  EXPECT_EQ(result_int->dtype, "int64");
  EXPECT_EQ(result_int->value, 1);

  // Test case for float
  auto result_float = freturn_primexpr(2.5).cast<TPrimExpr>();
  EXPECT_EQ(result_float->dtype, "float32");
  EXPECT_EQ(result_float->value, 2.5);

  // Test case for bool
  auto result_bool = freturn_primexpr(true).cast<TPrimExpr>();
  EXPECT_EQ(result_bool->dtype, "bool");
  EXPECT_EQ(result_bool->value, 1);

  // Test case for string
  auto result_string = freturn_primexpr("test_string").cast<TPrimExpr>();
  EXPECT_EQ(result_string->dtype, "test_string");
  EXPECT_EQ(result_string->value, 0);

  EXPECT_THROW(
      {
        try {
          freturn_primexpr(TInt(1));
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(
              error.message(),
              "Mismatched type on argument #0 when calling: `(0: test.PrimExpr) -> test.PrimExpr`. "
              "Expected `test.PrimExpr` but got `test.Int`");
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TVM_FFI_REGISTER_GLOBAL("testing.Int_GetValue").set_body_method(&TIntObj::GetValue);

TEST(Func, Register) {
  Function fget_value = Function::GetGlobalRequired("testing.Int_GetValue");
  TInt a(12);
  EXPECT_EQ(fget_value(a).cast<int>(), 12);
}
}  // namespace
