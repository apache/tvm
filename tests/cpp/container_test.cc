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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/container/variant.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace tvm;
using namespace tvm::tir;
using namespace tvm::runtime;

class TestErrorSwitch {
 public:
  // Need this so that destructor of temporary objects don't interrupt our
  // testing.
  TestErrorSwitch(const TestErrorSwitch& other) : should_fail(other.should_fail) {
    const_cast<TestErrorSwitch&>(other).should_fail = false;
  }

  explicit TestErrorSwitch(bool fail_flag) : should_fail{fail_flag} {}
  bool should_fail{false};

  ~TestErrorSwitch() {
    if (should_fail) {
      exit(1);
    }
  }
};

class TestArrayObj : public Object, public InplaceArrayBase<TestArrayObj, TestErrorSwitch> {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "test.TestArrayObj";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestArrayObj, Object);
  uint32_t size;

  size_t GetSize() const { return size; }

  template <typename Iterator>
  void Init(Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    this->size = 0;
    auto it = begin;
    for (size_t i = 0; i < num_elems; ++i) {
      InplaceArrayBase::EmplaceInit(i, *it++);
      if (i == 1) {
        throw std::bad_alloc();
      }
      // Only increment size after the initialization succeeds
      this->size++;
    }
  }

  template <typename Iterator>
  void WrongInit(Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    this->size = num_elems;
    auto it = begin;
    for (size_t i = 0; i < num_elems; ++i) {
      InplaceArrayBase::EmplaceInit(i, *it++);
      if (i == 1) {
        throw std::bad_alloc();
      }
    }
  }

  friend class InplaceArrayBase;
};

TEST(ADT, Constructor) {
  std::vector<ObjectRef> fields;
  auto f1 = ADT::Tuple(fields);
  auto f2 = ADT::Tuple(fields);
  ADT v1{1, {f1, f2}};
  ASSERT_EQ(f1.tag(), 0);
  ASSERT_EQ(f2.size(), 0);
  ASSERT_EQ(v1.tag(), 1);
  ASSERT_EQ(v1.size(), 2);
  ASSERT_EQ(Downcast<ADT>(v1[0]).tag(), 0);
  ASSERT_EQ(Downcast<ADT>(v1[1]).size(), 0);
}

TEST(InplaceArrayBase, BadExceptionSafety) {
  auto wrong_init = []() {
    TestErrorSwitch f1{false};
    // WrongInit will set size to 3 so it will call destructor at index 1, which
    // will exit with error status.
    TestErrorSwitch f2{true};
    TestErrorSwitch f3{false};
    std::vector<TestErrorSwitch> fields{f1, f2, f3};
    auto ptr = make_inplace_array_object<TestArrayObj, TestErrorSwitch>(fields.size());
    try {
      ptr->WrongInit(fields.begin(), fields.end());
    } catch (...) {
    }
    // Call ~InplaceArrayBase
    ptr.reset();
    // never reaches here.
    exit(0);
  };
  ASSERT_EXIT(wrong_init(), ::testing::ExitedWithCode(1), "");
}

TEST(InplaceArrayBase, ExceptionSafety) {
  auto correct_init = []() {
    TestErrorSwitch f1{false};
    // Init will fail at index 1, so destrucotr at index 1 should not be called
    // since it's not initalized.
    TestErrorSwitch f2{true};
    std::vector<TestErrorSwitch> fields{f1, f2};
    auto ptr = make_inplace_array_object<TestArrayObj, TestErrorSwitch>(fields.size());
    try {
      ptr->Init(fields.begin(), fields.end());
    } catch (...) {
    }
    // Call ~InplaceArrayBase
    ptr.reset();
    // Skip the destructors of f1, f2, and fields
    exit(0);
  };
  ASSERT_EXIT(correct_init(), ::testing::ExitedWithCode(0), "");
}

TEST(Array, PrimExpr) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Array<PrimExpr> list{x, z, z};
  LOG(INFO) << list.size();
  LOG(INFO) << list[0];
  LOG(INFO) << list[1];
}

TEST(Array, Mutate) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Array<PrimExpr> list{x, z, z};
  auto list2 = list;
  list.Set(1, x);
  ICHECK(list[1].same_as(x));
  ICHECK(list2[1].same_as(z));
}

TEST(Array, MutateInPlaceForUniqueReference) {
  using namespace tvm;
  Var x("x");
  Array<Var> arr{x, x};
  ICHECK(arr.unique());
  auto* before = arr.get();

  arr.MutateByApply([](Var) { return Var("y"); });
  auto* after = arr.get();
  ICHECK_EQ(before, after);
}

TEST(Array, CopyWhenMutatingNonUniqueReference) {
  using namespace tvm;
  Var x("x");
  Array<Var> arr{x, x};
  Array<Var> arr2 = arr;

  ICHECK(!arr.unique());
  auto* before = arr.get();

  arr.MutateByApply([](Var) { return Var("y"); });
  auto* after = arr.get();
  ICHECK_NE(before, after);
}

TEST(Array, Map) {
  // Basic functionality
  using namespace tvm;
  Var x("x");
  Var y("y");
  Array<Var> var_arr{x, y};
  Array<PrimExpr> expr_arr = var_arr.Map([](Var var) -> PrimExpr { return var + 1; });

  ICHECK_NE(var_arr.get(), expr_arr.get());
  ICHECK(expr_arr[0]->IsInstance<AddNode>());
  ICHECK(expr_arr[1]->IsInstance<AddNode>());
  ICHECK(expr_arr[0].as<AddNode>()->a.same_as(x));
  ICHECK(expr_arr[1].as<AddNode>()->a.same_as(y));
}

TEST(Array, MapToSameTypeWithoutCopy) {
  // If the applied map doesn't alter the contents, we can avoid a
  // copy.
  using namespace tvm;
  Var x("x");
  Var y("y");
  Array<Var> var_arr{x, y};
  Array<Var> var_arr2 = var_arr.Map([](Var var) { return var; });

  ICHECK_EQ(var_arr.get(), var_arr2.get());
}

TEST(Array, MapToSameTypeWithCopy) {
  // If the applied map does alter the contents, we need to make a
  // copy.  The loop in this test is to validate correct behavior
  // regardless of where the first discrepancy occurs.
  using namespace tvm;
  Var x("x");
  Var y("y");
  Var z("z");
  Var replacement("replacement");
  for (size_t i = 0; i < 2; i++) {
    Array<Var> var_arr{x, y, z};
    Var to_replace = var_arr[i];
    Array<Var> var_arr2 =
        var_arr.Map([&](Var var) { return var.same_as(to_replace) ? replacement : var; });

    ICHECK_NE(var_arr.get(), var_arr2.get());

    // The original array is unchanged
    ICHECK_EQ(var_arr.size(), 3);
    ICHECK(var_arr[0].same_as(x));
    ICHECK(var_arr[1].same_as(y));

    // The returned array has one of the elements replaced.
    ICHECK_EQ(var_arr2.size(), 3);
    ICHECK(var_arr2[i].same_as(replacement));
    ICHECK(i == 0 || var_arr2[0].same_as(x));
    ICHECK(i == 1 || var_arr2[1].same_as(y));
    ICHECK(i == 2 || var_arr2[2].same_as(z));
  }
}

TEST(Array, MapToSuperclassWithoutCopy) {
  // If a map is converting to a superclass, and the mapping function
  // array doesn't change the value other than a cast, we can avoid a
  // copy.
  using namespace tvm;
  Var x("x");
  Var y("y");
  Array<Var> var_arr{x, y};
  Array<PrimExpr> expr_arr = var_arr.Map([](Var var) { return PrimExpr(var); });

  ICHECK_EQ(var_arr.get(), expr_arr.get());
}

TEST(Array, MapToSubclassWithoutCopy) {
  // If a map is converting to a subclass, and the mapped array
  // happens to only contain instances of that subclass, we can
  // able to avoid a copy.
  using namespace tvm;
  Var x("x");
  Var y("y");
  Array<PrimExpr> expr_arr{x, y};
  Array<Var> var_arr = expr_arr.Map([](PrimExpr expr) -> Var { return Downcast<Var>(expr); });

  ICHECK_EQ(var_arr.get(), expr_arr.get());
}

TEST(Array, MapToOptionalWithoutCopy) {
  // Optional<T> and T both have the same T::ContainerType, just with
  // different interfaces for handling `T::data_ == nullptr`.
  using namespace tvm;
  Var x("x");
  Var y("y");
  Array<Var> var_arr{x, y};
  Array<Optional<Var>> opt_arr = var_arr.Map([](Var var) { return Optional<Var>(var); });

  ICHECK_EQ(var_arr.get(), opt_arr.get());
}

TEST(Array, MapFromOptionalWithoutCopy) {
  // Optional<T> and T both have the same T::ContainerType, just with
  // different interfaces for handling `T::data_ == nullptr`.
  using namespace tvm;
  Var x("x");
  Var y("y");
  Array<Optional<Var>> opt_arr{x, y};
  Array<Var> var_arr =
      opt_arr.Map([](Optional<Var> var) { return var.value_or(Var("undefined")); });

  ICHECK_EQ(var_arr.get(), opt_arr.get());
}

TEST(Array, Iterator) {
  using namespace tvm;
  Array<PrimExpr> array{1, 2, 3};
  std::vector<PrimExpr> vector(array.begin(), array.end());
  ICHECK(vector[1].as<IntImmNode>()->value == 2);
}

TEST(Array, PushPop) {
  using namespace tvm;
  Array<Integer> a;
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
  using namespace tvm;
  for (size_t n = 0; n < 10; ++n) {
    Array<Integer> a;
    Array<Integer> b;
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
  using namespace tvm;
  Array<Integer> a;
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
  using namespace tvm;
  Array<Integer> range_a{-1, -2, -3, -4};
  std::vector<int> range_b{-1, -2, -3, -4};
  Array<Integer> a;
  std::vector<int> b;
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

TEST(Map, Expr) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  auto zz = z + 1;
  Map<PrimExpr, PrimExpr> dict{{x, z}, {z, 2}};
  ICHECK(dict.size() == 2);
  ICHECK(dict[x].same_as(z));
  ICHECK(dict.count(z));
  ICHECK(!dict.count(zz));
}

TEST(Map, Str) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<String, PrimExpr> dict{{"x", z}, {"z", 2}};
  ICHECK(dict.size() == 2);
  ICHECK(dict["x"].same_as(z));
}

TEST(Map, Mutate) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<PrimExpr, PrimExpr> dict{{x, z}, {z, 2}};
  auto zz = z + 1;
  ICHECK(dict[x].same_as(z));
  dict.Set(x, zz);
  auto dict2 = dict;
  ICHECK(dict2.count(z) == 1);
  dict.Set(zz, x);
  ICHECK(dict2.count(zz) == 0);
  ICHECK(dict.count(zz) == 1);

  auto it = dict.find(zz);
  ICHECK(it != dict.end() && (*it).second.same_as(x));

  it = dict2.find(zz);
  ICHECK(it == dict2.end());
}

TEST(Map, Clear) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<PrimExpr, PrimExpr> dict{{x, z}, {z, 2}};
  ICHECK(dict.size() == 2);
  dict.clear();
  ICHECK(dict.size() == 0);
}

TEST(Map, Iterator) {
  using namespace tvm;
  PrimExpr a = 1, b = 2;
  Map<PrimExpr, PrimExpr> map1{{a, b}};
  std::unordered_map<PrimExpr, PrimExpr, ObjectPtrHash, ObjectPtrEqual> map2(map1.begin(),
                                                                             map1.end());
  ICHECK(map2[a].as<IntImmNode>()->value == 2);
}

TEST(Map, Insert) {
  using namespace tvm;
  auto check = [](const Map<String, Integer>& result,
                  std::unordered_map<std::string, int64_t> expected) {
    ICHECK_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      ICHECK(expected.count(kv.first));
      ICHECK_EQ(expected[kv.first], kv.second.IntValue());
      expected.erase(kv.first);
    }
  };
  Map<String, Integer> result;
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
  auto check = [](const Map<String, Integer>& result,
                  std::unordered_map<std::string, int64_t> expected) {
    ICHECK_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      ICHECK(expected.count(kv.first));
      ICHECK_EQ(expected[kv.first], kv.second.IntValue());
      expected.erase(kv.first);
    }
  };
  Map<String, Integer> map{{"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}, {"e", 5}};
  std::unordered_map<std::string, int64_t> stl;
  std::transform(map.begin(), map.end(), std::inserter(stl, stl.begin()),
                 [](auto&& p) { return std::make_pair(p.first, p.second.IntValue()); });
  for (char c = 'a'; c <= 'e'; ++c) {
    Map<String, Integer> result = map;
    std::unordered_map<std::string, int64_t> expected(stl);
    std::string key(1, c);
    result.erase(key);
    expected.erase(key);
    check(result, expected);
  }
}

#if TVM_LOG_DEBUG
TEST(Map, Race) {
  using namespace tvm::runtime;
  Map<Integer, Integer> m;

  m.Set(1, 1);
  Map<tvm::Integer, tvm::Integer>::iterator it = m.begin();
  EXPECT_NO_THROW({ auto& kv = *it; });

  m.Set(2, 2);
  // changed. iterator should be re-obtained
  EXPECT_ANY_THROW({ auto& kv = *it; });
}
#endif  // TVM_LOG_DEBUG

TEST(String, MoveFromStd) {
  using namespace std;
  string source = "this is a string";
  string expect = source;
  String s(std::move(source));
  string copy = (string)s;
  ICHECK_EQ(copy, expect);
  ICHECK_EQ(source.size(), 0);
}

TEST(String, CopyFromStd) {
  using namespace std;
  string source = "this is a string";
  string expect = source;
  String s{source};
  string copy = (string)s;
  ICHECK_EQ(copy, expect);
  ICHECK_EQ(source.size(), expect.size());
}

TEST(String, Assignment) {
  using namespace std;
  String s{string{"hello"}};
  s = string{"world"};
  ICHECK_EQ(s == "world", true);
  string s2{"world2"};
  s = std::move(s2);
  ICHECK_EQ(s == "world2", true);
}

TEST(String, empty) {
  using namespace std;
  String s{"hello"};
  ICHECK_EQ(s.empty(), false);
  s = std::string("");
  ICHECK_EQ(s.empty(), true);
}

TEST(String, Comparisons) {
  using namespace std;
  string source = "a string";
  string mismatch = "a string but longer";
  String s{source};
  String m{mismatch};

  ICHECK_EQ(s == source, true);
  ICHECK_EQ(s == mismatch, false);
  ICHECK_EQ(s == source.data(), true);
  ICHECK_EQ(s == mismatch.data(), false);

  ICHECK_EQ(s < m, source < mismatch);
  ICHECK_EQ(s > m, source > mismatch);
  ICHECK_EQ(s <= m, source <= mismatch);
  ICHECK_EQ(s >= m, source >= mismatch);
  ICHECK_EQ(s == m, source == mismatch);
  ICHECK_EQ(s != m, source != mismatch);

  ICHECK_EQ(m < s, mismatch < source);
  ICHECK_EQ(m > s, mismatch > source);
  ICHECK_EQ(m <= s, mismatch <= source);
  ICHECK_EQ(m >= s, mismatch >= source);
  ICHECK_EQ(m == s, mismatch == source);
  ICHECK_EQ(m != s, mismatch != source);
}

// Check '\0' handling
TEST(String, null_byte_handling) {
  using namespace std;
  // Ensure string still compares equal if it contains '\0'.
  string v1 = "hello world";
  size_t v1_size = v1.size();
  v1[5] = '\0';
  ICHECK_EQ(v1[5], '\0');
  ICHECK_EQ(v1.size(), v1_size);
  String str_v1{v1};
  ICHECK_EQ(str_v1.compare(v1), 0);
  ICHECK_EQ(str_v1.size(), v1_size);

  // Ensure bytes after '\0' are taken into account for mismatches.
  string v2 = "aaa one";
  string v3 = "aaa two";
  v2[3] = '\0';
  v3[3] = '\0';
  String str_v2{v2};
  String str_v3{v3};
  ICHECK_EQ(str_v2.compare(str_v3), -1);
  ICHECK_EQ(str_v2.size(), 7);
  // strcmp won't be able to detect the mismatch
  ICHECK_EQ(strcmp(v2.data(), v3.data()), 0);
  // string::compare can handle \0 since it knows size
  ICHECK_LT(v2.compare(v3), 0);

  // If there is mismatch before '\0', should still handle it.
  string v4 = "acc one";
  string v5 = "abb two";
  v4[3] = '\0';
  v5[3] = '\0';
  String str_v4{v4};
  String str_v5{v5};
  ICHECK_GT(str_v4.compare(str_v5), 0);
  ICHECK_EQ(str_v4.size(), 7);
  // strcmp is able to detect the mismatch
  ICHECK_GT(strcmp(v4.data(), v5.data()), 0);
  // string::compare can handle \0 since it knows size
  ICHECK_GT(v4.compare(v5), 0);
}

TEST(String, compare_same_memory_region_different_size) {
  using namespace std;
  string source = "a string";
  String str_source{source};
  char* memory = const_cast<char*>(str_source.data());
  ICHECK_EQ(str_source.compare(memory), 0);
  // This changes the string size
  memory[2] = '\0';
  // memory is logically shorter now
  ICHECK_GT(str_source.compare(memory), 0);
}

TEST(String, compare) {
  using namespace std;
  string source = "a string";
  string mismatch1 = "a string but longer";
  string mismatch2 = "a strin";
  string mismatch3 = "a b";
  string mismatch4 = "a t";
  String str_source{source};
  String str_mismatch1{mismatch1};
  String str_mismatch2{mismatch2};
  String str_mismatch3{mismatch3};
  String str_mismatch4{mismatch4};

  // compare with string
  ICHECK_EQ(str_source.compare(source), 0);
  ICHECK(str_source == source);
  ICHECK(source == str_source);
  ICHECK(str_source <= source);
  ICHECK(source <= str_source);
  ICHECK(str_source >= source);
  ICHECK(source >= str_source);
  ICHECK_LT(str_source.compare(mismatch1), 0);
  ICHECK(str_source < mismatch1);
  ICHECK(mismatch1 != str_source);
  ICHECK_GT(str_source.compare(mismatch2), 0);
  ICHECK(str_source > mismatch2);
  ICHECK(mismatch2 < str_source);
  ICHECK_GT(str_source.compare(mismatch3), 0);
  ICHECK(str_source > mismatch3);
  ICHECK_LT(str_source.compare(mismatch4), 0);
  ICHECK(str_source < mismatch4);
  ICHECK(mismatch4 > str_source);

  // compare with char*
  ICHECK_EQ(str_source.compare(source.data()), 0);
  ICHECK(str_source == source.data());
  ICHECK(source.data() == str_source);
  ICHECK(str_source <= source.data());
  ICHECK(source <= str_source.data());
  ICHECK(str_source >= source.data());
  ICHECK(source >= str_source.data());
  ICHECK_LT(str_source.compare(mismatch1.data()), 0);
  ICHECK(str_source < mismatch1.data());
  ICHECK(str_source != mismatch1.data());
  ICHECK(mismatch1.data() != str_source);
  ICHECK_GT(str_source.compare(mismatch2.data()), 0);
  ICHECK(str_source > mismatch2.data());
  ICHECK(mismatch2.data() < str_source);
  ICHECK_GT(str_source.compare(mismatch3.data()), 0);
  ICHECK(str_source > mismatch3.data());
  ICHECK_LT(str_source.compare(mismatch4.data()), 0);
  ICHECK(str_source < mismatch4.data());
  ICHECK(mismatch4.data() > str_source);

  // compare with String
  ICHECK_LT(str_source.compare(str_mismatch1), 0);
  ICHECK(str_source < str_mismatch1);
  ICHECK_GT(str_source.compare(str_mismatch2), 0);
  ICHECK(str_source > str_mismatch2);
  ICHECK_GT(str_source.compare(str_mismatch3), 0);
  ICHECK(str_source > str_mismatch3);
  ICHECK_LT(str_source.compare(str_mismatch4), 0);
  ICHECK(str_source < str_mismatch4);
}

TEST(String, c_str) {
  using namespace std;
  string source = "this is a string";
  string mismatch = "mismatch";
  String s{source};

  ICHECK_EQ(std::strcmp(s.c_str(), source.data()), 0);
  ICHECK_NE(std::strcmp(s.c_str(), mismatch.data()), 0);
}

TEST(String, hash) {
  using namespace std;
  string source = "this is a string";
  String s{source};
  std::hash<String>()(s);

  std::unordered_map<String, std::string> map;
  String k1{string{"k1"}};
  string v1{"v1"};
  String k2{string{"k2"}};
  string v2{"v2"};
  map[k1] = v1;
  map[k2] = v2;

  ICHECK_EQ(map[k1], v1);
  ICHECK_EQ(map[k2], v2);
}

TEST(String, Cast) {
  using namespace std;
  string source = "this is a string";
  String s{source};
  ObjectRef r = s;
  String s2 = Downcast<String>(r);
}

TEST(String, Concat) {
  String s1("hello");
  String s2("world");
  std::string s3("world");
  String res1 = s1 + s2;
  String res2 = s1 + s3;
  String res3 = s3 + s1;
  String res4 = s1 + "world";
  String res5 = "world" + s1;

  ICHECK_EQ(res1.compare("helloworld"), 0);
  ICHECK_EQ(res2.compare("helloworld"), 0);
  ICHECK_EQ(res3.compare("worldhello"), 0);
  ICHECK_EQ(res4.compare("helloworld"), 0);
  ICHECK_EQ(res5.compare("worldhello"), 0);
}

TEST(Optional, Composition) {
  Optional<String> opt0(nullptr);
  Optional<String> opt1 = String("xyz");
  Optional<String> opt2 = String("xyz1");
  // operator bool
  ICHECK(!opt0);
  ICHECK(opt1);
  // comparison op
  ICHECK(opt0 != "xyz");
  ICHECK(opt1 == "xyz");
  ICHECK(opt1 != nullptr);
  ICHECK(opt0 == nullptr);
  ICHECK(opt0.value_or("abc") == "abc");
  ICHECK(opt1.value_or("abc") == "xyz");
  ICHECK(opt0 != opt1);
  ICHECK(opt1 == Optional<String>(String("xyz")));
  ICHECK(opt0 == Optional<String>(nullptr));
  opt0 = opt1;
  ICHECK(opt0 == opt1);
  ICHECK(opt0.value().same_as(opt1.value()));
  opt0 = std::move(opt2);
  ICHECK(opt0 != opt2);
}

TEST(Optional, IntCmp) {
  Integer val(CallingConv::kDefault);
  Optional<Integer> opt = Integer(0);
  ICHECK(0 == static_cast<int>(CallingConv::kDefault));
  ICHECK(val == CallingConv::kDefault);
  ICHECK(opt == CallingConv::kDefault);

  // check we can handle implicit 0 to nullptr conversion.
  Optional<Integer> opt1(nullptr);
  ICHECK(opt1 != 0);
  ICHECK(opt1 != false);
  ICHECK(!(opt1 == 0));
}

TEST(Optional, PackedCall) {
  auto tf = [](Optional<String> s, bool isnull) {
    if (isnull) {
      ICHECK(s == nullptr);
    } else {
      ICHECK(s != nullptr);
    }
    return s;
  };
  auto func = TypedPackedFunc<Optional<String>(Optional<String>, bool)>(tf);
  ICHECK(func(String("xyz"), false) == "xyz");
  ICHECK(func(Optional<String>(nullptr), true) == nullptr);

  auto pf = [](TVMArgs args, TVMRetValue* rv) {
    Optional<String> s = args[0];
    bool isnull = args[1];
    if (isnull) {
      ICHECK(s == nullptr);
    } else {
      ICHECK(s != nullptr);
    }
    *rv = s;
  };
  auto packedfunc = PackedFunc(pf);
  ICHECK(packedfunc("xyz", false).operator String() == "xyz");
  ICHECK(packedfunc("xyz", false).operator Optional<String>() == "xyz");
  ICHECK(packedfunc(nullptr, true).operator Optional<String>() == nullptr);

  // test FFI convention.
  auto test_ffi = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
    int tcode = args[1];
    ICHECK_EQ(args[0].type_code(), tcode);
  });
  String s = "xyz";
  auto nd = NDArray::Empty({0, 1}, DataType::Float(32), DLDevice{kDLCPU, 0});
  test_ffi(Optional<NDArray>(nd), static_cast<int>(kTVMNDArrayHandle));
  test_ffi(Optional<String>(s), static_cast<int>(kTVMObjectRValueRefArg));
  test_ffi(s, static_cast<int>(kTVMObjectHandle));
  test_ffi(String(s), static_cast<int>(kTVMObjectRValueRefArg));
}

TEST(Variant, Construct) {
  Variant<PrimExpr, String> variant;
  variant = PrimExpr(1);
  ICHECK(variant.as<PrimExpr>());
  ICHECK(!variant.as<String>());

  variant = String("hello");
  ICHECK(variant.as<String>());
  ICHECK(!variant.as<PrimExpr>());
}

TEST(Variant, InvalidTypeThrowsError) {
  auto expected_to_throw = []() {
    ObjectPtr<Object> node = make_object<Object>();
    Variant<PrimExpr, String> variant(node);
  };

  EXPECT_THROW(expected_to_throw(), InternalError);
}

TEST(Variant, ReferenceIdentifyPreservedThroughAssignment) {
  Variant<PrimExpr, String> variant;
  ICHECK(!variant.defined());

  String string_obj = "dummy_test";
  variant = string_obj;
  ICHECK(variant.defined());
  ICHECK(variant.same_as(string_obj));
  ICHECK(string_obj.same_as(variant));

  String out_string_obj = Downcast<String>(variant);
  ICHECK(string_obj.same_as(out_string_obj));
}

TEST(Variant, ExtractValueFromAssignment) {
  Variant<PrimExpr, String> variant = String("hello");
  ICHECK_EQ(variant.as<String>().value(), "hello");
}

TEST(Variant, AssignmentFromVariant) {
  Variant<PrimExpr, String> variant = String("hello");
  auto variant2 = variant;
  ICHECK(variant2.as<String>());
  ICHECK_EQ(variant2.as<String>().value(), "hello");
}
