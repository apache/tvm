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
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Map, Basic) {
  Map<TInt, int> map0;
  TInt k0(0);
  map0.Set(k0, 1);

  EXPECT_EQ(map0.size(), 1);

  map0.Set(k0, 2);
  EXPECT_EQ(map0.size(), 1);

  auto it = map0.find(k0);
  EXPECT_TRUE(it != map0.end());
  EXPECT_EQ((*it).second, 2);
}

TEST(Map, PODKey) {
  Map<Any, Any> map0;

  // int as key
  map0.Set(1, 2);
  // float key is different
  map0.Set(1.1, 3);
  EXPECT_EQ(map0.size(), 2);

  auto it = map0.find(1.1);
  EXPECT_TRUE(it != map0.end());
  EXPECT_EQ((*it).second.operator int(), 3);
}

TEST(Map, Object) {
  TInt x(1);
  TInt z(100);
  TInt zz(1000);
  Map<TInt, TInt> dict{{x, z}, {z, zz}};
  EXPECT_EQ(dict.size(), 2);
  EXPECT_TRUE(dict[x].same_as(z));
  EXPECT_TRUE(dict.count(z));
  EXPECT_TRUE(!dict.count(zz));
}

TEST(Map, Str) {
  TInt x(1);
  TInt z(100);
  Map<String, TInt> dict{{"x", z}, {"z", z}};
  EXPECT_EQ(dict.size(), 2);
  EXPECT_TRUE(dict["x"].same_as(z));
}

TEST(Map, Mutate) {
  TInt x(1);
  TInt z(100);
  TInt zz(1000);
  Map<TInt, TInt> dict{{x, z}, {z, zz}};

  EXPECT_TRUE(dict[x].same_as(z));
  dict.Set(x, zz);
  auto dict2 = dict;
  EXPECT_EQ(dict2.count(z), 1);
  dict.Set(zz, x);
  EXPECT_EQ(dict2.count(zz), 0);
  EXPECT_EQ(dict.count(zz), 1);

  auto it = dict.find(zz);
  EXPECT_TRUE(it != dict.end() && (*it).second.same_as(x));

  it = dict2.find(zz);
  EXPECT_TRUE(it == dict2.end());
}

TEST(Map, Clear) {
  TInt x(1);
  TInt z(100);
  Map<TInt, TInt> dict{{x, z}, {z, z}};
  EXPECT_EQ(dict.size(), 2);
  dict.clear();
  EXPECT_EQ(dict.size(), 0);
}

TEST(Map, Insert) {
  auto check = [](const Map<String, int64_t>& result,
                  std::unordered_map<std::string, int64_t> expected) {
    EXPECT_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      EXPECT_TRUE(expected.count(kv.first));
      EXPECT_EQ(expected[kv.first], kv.second);
      expected.erase(kv.first);
    }
  };
  Map<String, int64_t> result;
  std::unordered_map<std::string, int64_t> expected;
  char key = 'a';
  int64_t val = 1;
  for (int i = 0; i < 26; ++i, ++key, ++val) {
    std::string s(1, key);
    result.Set(s, val);
    expected[s] = val;
    check(result, expected);
  }
}

TEST(Map, Erase) {
  auto check = [](const Map<String, int64_t>& result,
                  std::unordered_map<std::string, int64_t> expected) {
    EXPECT_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      EXPECT_TRUE(expected.count(kv.first));
      EXPECT_EQ(expected[kv.first], kv.second);
      expected.erase(kv.first);
    }
  };
  Map<String, int64_t> map{{"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}, {"e", 5}};
  std::unordered_map<std::string, int64_t> stl;
  std::transform(map.begin(), map.end(), std::inserter(stl, stl.begin()),
                 [](auto&& p) { return std::make_pair(p.first, p.second); });
  for (char c = 'a'; c <= 'e'; ++c) {
    Map<String, int64_t> result = map;
    std::unordered_map<std::string, int64_t> expected(stl);
    std::string key(1, c);
    result.erase(key);
    expected.erase(key);
    check(result, expected);
  }
}

TEST(Map, AnyImplicitConversion) {
  Map<Any, Any> map0;
  map0.Set(1, 2);
  map0.Set(2, 3.1);
  EXPECT_EQ(map0.size(), 2);

  // check will trigger copy
  AnyView view0 = map0;
  Map<int, double> map1 = view0;
  EXPECT_TRUE(!map1.same_as(map0));
  EXPECT_EQ(map1[1], 2);
  EXPECT_EQ(map1[2], 3.1);
  EXPECT_EQ(map1.use_count(), 1);

  Map<int, Any> map2 = view0;
  EXPECT_TRUE(map2.same_as(map0));
  EXPECT_EQ(map2.use_count(), 2);

  Map<Any, double> map3 = view0;
  EXPECT_TRUE(!map3.same_as(map0));
  EXPECT_EQ(map3.use_count(), 1);

  Map<Any, Any> map4{{"yes", 1.1}, {"no", 2.2}};
  Any any1 = map4;

  Map<String, double> map5 = any1;
  EXPECT_TRUE(map5.same_as(map4));
  EXPECT_EQ(map5.use_count(), 3);

  Map<String, Any> map6 = any1;
  EXPECT_TRUE(map6.same_as(map4));
  EXPECT_EQ(map6.use_count(), 4);

  EXPECT_EQ(map6["yes"].operator double(), 1.1);
  EXPECT_EQ(map6["no"].operator double(), 2.2);

  Map<Any, Any> map7 = any1;
  EXPECT_TRUE(map7.same_as(map4));
  EXPECT_EQ(map7.use_count(), 5);

  Map<Any, TPrimExpr> map8 = any1;
  EXPECT_TRUE(!map8.same_as(map4));
  EXPECT_EQ(map8.use_count(), 1);
  EXPECT_EQ(map8["yes"]->value, 1.1);
  EXPECT_EQ(map8["no"]->value, 2.2);
}

TEST(Map, AnyConvertCheck) {
  Map<Any, Any> map = {{11, 1.1}};
  EXPECT_EQ(map[11].operator double(), 1.1);

  AnyView view0 = map;
  Map<int64_t, double> arr1 = view0;
  EXPECT_EQ(arr1[11], 1.1);

  Any any1 = map;
  using WrongMap = Map<int64_t, int64_t>;

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] WrongMap arr2 = any1;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(
              what.find(
                  "Cannot convert from type `Map[K, some value is float]` to `Map<int, int>`"),
              std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  using WrongMap2 = Map<TNumber, double>;
  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] WrongMap2 arr2 = any1;
        } catch (const Error& error) {
          EXPECT_EQ(error->kind, "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `Map[some key is int, V]` to "
                              "`Map<test.Number, float>`"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(Map, PackedFuncGetItem) {
  Function f = Function::FromUnpacked(
      [](const MapObj* n, const Any& k) -> Any { return n->at(k); }, "map_get_item");
  Map<String, int64_t> map{{"x", 1}, {"y", 2}};
  Any k("x");
  Any v = f(map, k);
  EXPECT_EQ(v.operator int(), 1);
}

TEST(Map, Upcast) {
  Map<int, int> m0 = {{1, 2}, {3, 4}};
  Map<Any, Any> m1 = m0;
  EXPECT_EQ(m1[1].operator int(), 2);
  EXPECT_EQ(m1[3].operator int(), 4);
  static_assert(details::type_contains_v<Map<Any, Any>, Map<String, int>>);

  Map<String, Array<int>> m2 = {{"x", {1}}, {"y", {2}}};
  Map<String, Array<Any>> m3 = m2;
}

}  // namespace
