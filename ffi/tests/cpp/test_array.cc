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
#include <tvm/ffi/container/array.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Array, Basic) {
  Array<TInt> arr = {TInt(11), TInt(12)};
  TInt v1 = arr[0];
  EXPECT_EQ(v1->value, 11);
  EXPECT_EQ(v1.use_count(), 2);
  EXPECT_EQ(arr[1]->value, 12);
}

TEST(Array, COWSet) {
  Array<TInt> arr = {TInt(11), TInt(12)};
  Array<TInt> arr2 = arr;
  EXPECT_EQ(arr.use_count(), 2);
  arr.Set(1, TInt(13));
  EXPECT_EQ(arr.use_count(), 1);
  EXPECT_EQ(arr[1]->value, 13);
  EXPECT_EQ(arr2[1]->value, 12);
}

TEST(Array, MutateInPlaceForUniqueReference) {
  TInt x(1);
  Array<TInt> arr{x, x};
  EXPECT_TRUE(arr.unique());
  auto* before = arr.get();

  arr.MutateByApply([](TInt) { return TInt(2); });
  auto* after = arr.get();
  EXPECT_EQ(before, after);
}

TEST(Array, CopyWhenMutatingNonUniqueReference) {
  TInt x(1);
  Array<TInt> arr{x, x};
  Array<TInt> arr2 = arr;

  EXPECT_TRUE(!arr.unique());
  auto* before = arr.get();

  arr.MutateByApply([](TInt) { return TInt(2); });
  auto* after = arr.get();
  EXPECT_NE(before, after);
}

TEST(Array, Map) {
  // Basic functionality
  TInt x(1), y(1);
  Array<TInt> var_arr{x, y};
  Array<TNumber> expr_arr = var_arr.Map([](TInt var) -> TNumber { return TFloat(var->value + 1); });

  EXPECT_NE(var_arr.get(), expr_arr.get());
  EXPECT_TRUE(expr_arr[0]->IsInstance<TFloatObj>());
  EXPECT_TRUE(expr_arr[1]->IsInstance<TFloatObj>());
}

TEST(Array, Iterator) {
  Array<int> array{1, 2, 3};
  std::vector<int> vector(array.begin(), array.end());
  EXPECT_EQ(vector[1], 2);
}

TEST(Array, PushPop) {
  Array<int> a;
  std::vector<int> b;
  for (int i = 0; i < 10; ++i) {
    a.push_back(i);
    b.push_back(i);
    ASSERT_EQ(a.front(), b.front());
    ASSERT_EQ(a.back(), b.back());
    ASSERT_EQ(a.size(), b.size());
    int n = a.size();
    for (int j = 0; j < n; ++j) {
      ASSERT_EQ(a[j], b[j]);
    }
  }
  for (int i = 9; i >= 0; --i) {
    ASSERT_EQ(a.front(), b.front());
    ASSERT_EQ(a.back(), b.back());
    ASSERT_EQ(a.size(), b.size());
    a.pop_back();
    b.pop_back();
    int n = a.size();
    for (int j = 0; j < n; ++j) {
      ASSERT_EQ(a[j], b[j]);
    }
  }
  ASSERT_EQ(a.empty(), true);
}

TEST(Array, ResizeReserveClear) {
  for (size_t n = 0; n < 10; ++n) {
    Array<int> a;
    Array<int> b;
    a.resize(n);
    b.reserve(n);
    ASSERT_EQ(a.size(), n);
    ASSERT_GE(a.capacity(), n);
    a.clear();
    b.clear();
    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(b.size(), 0);
  }
}

TEST(Array, InsertErase) {
  Array<int> a;
  std::vector<int> b;
  for (int n = 1; n <= 10; ++n) {
    a.insert(a.end(), n);
    b.insert(b.end(), n);
    for (int pos = 0; pos <= n; ++pos) {
      a.insert(a.begin() + pos, pos);
      b.insert(b.begin() + pos, pos);
      ASSERT_EQ(a.front(), b.front());
      ASSERT_EQ(a.back(), b.back());
      ASSERT_EQ(a.size(), n + 1);
      ASSERT_EQ(b.size(), n + 1);
      for (int k = 0; k <= n; ++k) {
        ASSERT_EQ(a[k], b[k]);
      }
      a.erase(a.begin() + pos);
      b.erase(b.begin() + pos);
    }
    ASSERT_EQ(a.front(), b.front());
    ASSERT_EQ(a.back(), b.back());
    ASSERT_EQ(a.size(), n);
  }
}

TEST(Array, InsertEraseRange) {
  Array<int> range_a{-1, -2, -3, -4};
  std::vector<int> range_b{-1, -2, -3, -4};
  Array<int> a;
  std::vector<int> b;

  static_assert(std::is_same_v<decltype(*range_a.begin()), int>);
  for (size_t n = 1; n <= 10; ++n) {
    a.insert(a.end(), n);
    b.insert(b.end(), n);
    for (size_t pos = 0; pos <= n; ++pos) {
      a.insert(a.begin() + pos, range_a.begin(), range_a.end());
      b.insert(b.begin() + pos, range_b.begin(), range_b.end());
      ASSERT_EQ(a.front(), b.front());
      ASSERT_EQ(a.back(), b.back());
      ASSERT_EQ(a.size(), n + range_a.size());
      ASSERT_EQ(b.size(), n + range_b.size());
      size_t m = n + range_a.size();
      for (size_t k = 0; k < m; ++k) {
        ASSERT_EQ(a[k], b[k]);
      }
      a.erase(a.begin() + pos, a.begin() + pos + range_a.size());
      b.erase(b.begin() + pos, b.begin() + pos + range_b.size());
    }
    ASSERT_EQ(a.front(), b.front());
    ASSERT_EQ(a.back(), b.back());
    ASSERT_EQ(a.size(), n);
  }
}

TEST(Array, FuncArrayAnyArg) {
  Function fadd_one =
      Function::FromUnpacked([](Array<Any> a) -> Any { return a[0].operator int() + 1; });
  EXPECT_EQ(fadd_one(Array<Any>{1}).operator int(), 2);
}

TEST(Array, MapUniquePropogation) {
  // Basic functionality
  Array<TInt> var_arr{TInt(1), TInt(2)};
  var_arr.MutateByApply([](TInt x) -> TInt {
    EXPECT_TRUE(x.unique());
    return x;
  });
}

TEST(Array, AnyImplicitConversion) {
  Array<Any> arr0_mixed = {11.1, 1};
  EXPECT_EQ(arr0_mixed[1].operator int(), 1);

  AnyView view0 = arr0_mixed;
  Array<double> arr0_float = view0;
  // they are not the same because arr_mixed
  // stores arr_mixed[1] as int but we need to convert to float
  EXPECT_TRUE(!arr0_float.same_as(arr0_mixed));
  EXPECT_EQ(arr0_float[1], 1.0);

  Any any1 = arr0_float;
  // if storage check passes, the same array get returned
  Array<double> arr1_float = any1;
  EXPECT_TRUE(arr1_float.same_as(arr0_float));
  // total count equals 3 include any1
  EXPECT_EQ(arr1_float.use_count(), 3);

  // convert to Array<Any> do not need any conversion
  Array<Any> arr1_mixed = any1;
  EXPECT_TRUE(arr1_mixed.same_as(arr1_float));
  EXPECT_EQ(arr1_float.use_count(), 4);
}

TEST(Array, AnyConvertCheck) {
  Array<Any> arr = {11.1, 1};
  EXPECT_EQ(arr[1].operator int(), 1);

  AnyView view0 = arr;
  Array<double> arr1 = view0;
  EXPECT_EQ(arr1[0], 11.1);
  EXPECT_EQ(arr1[1], 1.0);

  Any any1 = arr;

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] Array<int> arr2 = any1;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `Array[index 0: float]` to `Array<int>`"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  Array<Array<TNumber>> arr_nested = {{}, {TInt(1), TFloat(2)}};
  any1 = arr_nested;
  Array<Array<TNumber>> arr1_nested = any1;
  EXPECT_EQ(arr1_nested.use_count(), 3);

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] Array<Array<int>> arr2 = any1;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("`Array[index 1: Array[index 0: test.Int]]` to `Array<Array<int>>`"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(Array, Upcast) {
  Array<int> a0 = {1, 2, 3};
  Array<Any> a1 = a0;
  EXPECT_EQ(a1[0].operator int(), 1);
  EXPECT_EQ(a1[1].operator int(), 2);
  EXPECT_EQ(a1[2].operator int(), 3);

  Array<Array<int>> a2 = {a0};
  Array<Array<Any>> a3 = a2;
  Array<Array<Any>> a4 = a2;

  static_assert(details::type_contains_v<Array<Any>, Array<int>>);
  static_assert(details::type_contains_v<Any, Array<float>>);
}

}  // namespace
