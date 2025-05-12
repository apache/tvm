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
  EXPECT_EQ((*it).second.cast<int>(), 3);
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
  auto map1 = view0.cast<Map<int, double>>();
  EXPECT_TRUE(!map1.same_as(map0));
  EXPECT_EQ(map1[1], 2);
  EXPECT_EQ(map1[2], 3.1);
  EXPECT_EQ(map1.use_count(), 1);

  auto map2 = view0.cast<Map<int, Any>>();
  EXPECT_TRUE(map2.same_as(map0));
  EXPECT_EQ(map2.use_count(), 2);

  auto map3 = view0.cast<Map<Any, double>>();
  EXPECT_TRUE(!map3.same_as(map0));
  EXPECT_EQ(map3.use_count(), 1);

  Map<Any, Any> map4{{"yes", 1.1}, {"no", 2.2}};
  Any any1 = map4;

  auto map5 = any1.cast<Map<String, double>>();
  EXPECT_TRUE(map5.same_as(map4));
  EXPECT_EQ(map5.use_count(), 3);

  auto map6 = any1.cast<Map<String, Any>>();
  EXPECT_TRUE(map6.same_as(map4));
  EXPECT_EQ(map6.use_count(), 4);

  EXPECT_EQ(map6["yes"].cast<double>(), 1.1);
  EXPECT_EQ(map6["no"].cast<double>(), 2.2);

  auto map7 = any1.cast<Map<Any, Any>>();
  EXPECT_TRUE(map7.same_as(map4));
  EXPECT_EQ(map7.use_count(), 5);

  auto map8 = any1.cast<Map<Any, TPrimExpr>>();
  EXPECT_TRUE(!map8.same_as(map4));
  EXPECT_EQ(map8.use_count(), 1);
  EXPECT_EQ(map8["yes"]->value, 1.1);
  EXPECT_EQ(map8["no"]->value, 2.2);
}

TEST(Map, AnyConvertCheck) {
  Map<Any, Any> map = {{11, 1.1}};
  EXPECT_EQ(map[11].cast<double>(), 1.1);

  AnyView view0 = map;
  auto arr1 = view0.cast<Map<int64_t, double>>();
  EXPECT_EQ(arr1[11], 1.1);

  Any any1 = map;
  using WrongMap = Map<int64_t, int64_t>;

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto arr2 = any1.cast<WrongMap>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
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
          [[maybe_unused]] auto arr2 = any1.cast<WrongMap2>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot convert from type `Map[some key is int, V]` to "
                              "`Map<test.Number, float>`"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(Map, FunctionGetItem) {
  Function f = Function::FromTyped([](const MapObj* n, const Any& k) -> Any { return n->at(k); },
                                   "map_get_item");
  Map<String, int64_t> map{{"x", 1}, {"y", 2}};
  Any k("x");
  Any v = f(map, k);
  EXPECT_EQ(v.cast<int>(), 1);
}

TEST(Map, Upcast) {
  Map<int, int> m0 = {{1, 2}, {3, 4}};
  Map<Any, Any> m1 = m0;
  EXPECT_EQ(m1[1].cast<int>(), 2);
  EXPECT_EQ(m1[3].cast<int>(), 4);
  static_assert(details::type_contains_v<Map<Any, Any>, Map<String, int>>);

  Map<String, Array<int>> m2 = {{"x", {1}}, {"y", {2}}};
  Map<String, Array<Any>> m3 = m2;
}

template <typename K, typename V>
void PrintMap(const Map<K, V>& m0) {
  std::cout << "{";
  for (auto it = m0.begin(); it != m0.end(); ++it) {
    if (it != m0.begin()) {
      std::cout << ", ";
    }
    std::cout << (*it).first << ": " << (*it).second;
  }
  std::cout << "}" << std::endl;
}

TEST(Map, MapInsertOrder) {
  // test that map preserves the insertion order
  auto get_reverse_order = [](size_t size) {
    std::vector<int> reverse_order;
    for (int i = static_cast<int>(size); i != 0; --i) {
      reverse_order.push_back(i - 1);
    }
    return reverse_order;
  };

  auto check_map = [&](Map<String, int> m0, size_t size, const std::vector<int>& order) {
    auto lhs = m0.begin();
    auto rhs = order.begin();
    while (lhs != m0.end()) {
      TVM_FFI_ICHECK_EQ((*lhs).first, "hello" + std::to_string(*rhs));
      TVM_FFI_ICHECK_EQ((*lhs).second, *rhs);
      ++lhs;
      ++rhs;
    }
    lhs = m0.end();
    rhs = order.begin() + size;
    do {
      --lhs;
      --rhs;
      TVM_FFI_ICHECK_EQ((*lhs).first, "hello" + std::to_string(*rhs));
      TVM_FFI_ICHECK_EQ((*lhs).second, *rhs);
    } while (lhs != m0.begin());
  };

  auto check_order = [&](std::vector<int> order) {
    Map<String, int> m0;
    for (size_t i = 0; i < order.size(); ++i) {
      m0.Set("hello" + std::to_string(order[i]), order[i]);
      check_map(m0, i + 1, order);
    }
    check_map(m0, order.size(), order);
    // erase a few items
    m0.erase("hello" + std::to_string(order[0]));
    auto item0 = order[0];
    order.erase(order.begin());
    check_map(m0, order.size(), order);
    // erase the middle part
    if (order.size() > 1) {
      m0.erase("hello" + std::to_string(order[1]));
      order.erase(order.begin() + 1);
      check_map(m0, order.size(), order);
    }
    // erase the end
    m0.erase("hello" + std::to_string(order.back()));
    auto item2 = order.back();
    order.erase(order.end() - 1);
    check_map(m0, order.size(), order);
    EXPECT_NE(m0.size(), 0);
    // put back some items
    order.push_back(item2);
    m0.Set("hello" + std::to_string(item2), item2);
    check_map(m0, order.size(), order);
    order.push_back(item0);
    m0.Set("hello" + std::to_string(item0), item0);
    check_map(m0, order.size(), order);
  };
  // test with 17 items: DenseMapObj
  check_order(get_reverse_order(17));
  // test with 4 items: SmallMapObj
  check_order(get_reverse_order(4));
}

TEST(Map, EmptyIter) {
  Map<String, int> m0;
  EXPECT_EQ(m0.begin(), m0.end());
  // create a big map and then erase to keep a dense map empty
  for (int i = 0; i < 10; ++i) {
    m0.Set("hello" + std::to_string(i), i);
  }
  for (int i = 0; i < 10; ++i) {
    m0.erase("hello" + std::to_string(i));
  }
  EXPECT_EQ(m0.size(), 0);
  // now m0 is dense map with all empty slots
  EXPECT_EQ(m0.begin(), m0.end());
}
}  // namespace
