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
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/memory.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Variant, Basic) {
  Variant<int, float> v1 = 1;
  EXPECT_EQ(v1.get<int>(), 1);

  Variant<int, float> v2 = 2.0f;
  EXPECT_EQ(v2.get<float>(), 2.0f);
  v2 = v1;
  EXPECT_EQ(v2.get<int>(), 1);
}

TEST(Variant, AnyConvert) {
  Variant<int, TInt> v = 1;
  AnyView view0 = v;
  EXPECT_EQ(view0.as<int>().value(), 1);

  // implicit convert to variant
  Any any0 = 1;
  auto v1 = any0.cast<Variant<TPrimExpr, Array<TPrimExpr>>>();
  EXPECT_EQ(v1.get<TPrimExpr>()->value, 1);

  // move from any to variant
  Variant<TInt, int> v2 = TInt(1);
  Any any1 = std::move(v2);
  auto v3 = std::move(any1).cast<Variant<TInt, int>>();
  auto v4 = std::move(v3).get<TInt>();
  EXPECT_EQ(v4->value, 1);
  EXPECT_EQ(v4.use_count(), 1);
}

TEST(Variant, ObjectPtrHashEqual) {
  TInt x = TInt(1);
  TFloat y = TFloat(1.0f);

  Variant<TFloat, TInt> v0 = x;
  Variant<TFloat, TInt> v1 = y;
  Variant<TFloat, TInt> v2 = v1;

  EXPECT_EQ(ObjectPtrHash()(v0), ObjectPtrHash()(x));
  EXPECT_TRUE(!ObjectPtrEqual()(v0, v1));
  EXPECT_TRUE(!ObjectPtrEqual()(v0, v2));
}

TEST(Variant, FromTyped) {
  // try decution
  Function fadd1 = Function::FromTyped([](const Variant<int, TInt>& a) -> int64_t {
    if (auto opt_int = a.as<int>()) {
      return opt_int.value() + 1;
    } else {
      return a.get<TInt>()->value + 1;
    }
  });
  int b = fadd1(1).cast<int>();
  EXPECT_EQ(b, 2);

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd1(1.1);
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(
              error.message(),
              "Mismatched type on argument #0 when calling: `(0: Variant<int, test.Int>) -> int`. "
              "Expected `Variant<int, test.Int>` but got `float`");
          throw;
        }
      },
      ::tvm::ffi::Error);

  Function fadd2 = Function::FromTyped([](const Array<Variant<int, TInt>>& a) -> int64_t {
    if (auto opt_int = a[0].as<int>()) {
      return opt_int.value() + 1;
    } else {
      return a[0].get<TInt>()->value + 1;
    }
  });
  int c = fadd2(Array<Any>({1, 2})).cast<int>();
  EXPECT_EQ(c, 2);

  // convert that triggers error
  EXPECT_THROW(
      {
        try {
          fadd2(Array<Any>({1, 1.1}));
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          EXPECT_EQ(error.message(),
                    "Mismatched type on argument #0 when calling: `(0: Array<Variant<int, "
                    "test.Int>>) -> int`. "
                    "Expected `Array<Variant<int, test.Int>>` but got `Array[index 1: float]`");
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(Variant, Upcast) {
  Array<int> a0 = {1, 2, 3};
  static_assert(details::type_contains_v<Array<Variant<int, float>>, Array<int>>);
  Array<Variant<int, float>> a1 = a0;
  EXPECT_EQ(a1[0].get<int>(), 1);
}

TEST(Variant, AllObjectRef) {
  Variant<TInt, Array<TInt>> v0 = TInt(1);
  EXPECT_EQ(v0.get<TInt>()->value, 1);
  static_assert(std::is_base_of_v<ObjectRef, decltype(v0)>);
  Any any0 = v0;
  EXPECT_EQ(any0.cast<TInt>()->value, 1);
  auto v2 = any0.cast<Variant<TInt, Array<TInt>>>();
  EXPECT_TRUE(v0.same_as(v2));
  // assignment operator
  v0 = Array<TInt>({TInt(2), TInt(3)});
  EXPECT_EQ(v0.get<Array<TInt>>().size(), 2);
  EXPECT_EQ(v0.get<Array<TInt>>()[0]->value, 2);
  EXPECT_EQ(v0.get<Array<TInt>>()[1]->value, 3);
  EXPECT_EQ(sizeof(v0), sizeof(ObjectRef));
}

TEST(Variant, PODSameAs) {
  Variant<String, int> v0 = 1;
  Variant<String, int> v1 = 1;
  EXPECT_TRUE(v0.same_as(v1));
  String s = String("hello");
  v0 = s;
  v1 = s;
  EXPECT_TRUE(v0.same_as(v1));
  v1 = String("hello");
  EXPECT_TRUE(!v0.same_as(v1));
}
}  // namespace
