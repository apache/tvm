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
#include <tvm/runtime/container.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include <new>
#include <unordered_map>
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

  TestErrorSwitch(bool fail_flag) : should_fail{fail_flag} {}
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
  CHECK(list[1].same_as(x));
  CHECK(list2[1].same_as(z));
}

TEST(Array, Iterator) {
  using namespace tvm;
  Array<PrimExpr> array{1, 2, 3};
  std::vector<PrimExpr> vector(array.begin(), array.end());
  CHECK(vector[1].as<IntImmNode>()->value == 2);
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
  CHECK(dict.size() == 2);
  CHECK(dict[x].same_as(z));
  CHECK(dict.count(z));
  CHECK(!dict.count(zz));
}

TEST(Map, Str) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<String, PrimExpr> dict{{"x", z}, {"z", 2}};
  CHECK(dict.size() == 2);
  CHECK(dict["x"].same_as(z));
}

TEST(Map, Mutate) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<PrimExpr, PrimExpr> dict{{x, z}, {z, 2}};
  auto zz = z + 1;
  CHECK(dict[x].same_as(z));
  dict.Set(x, zz);
  auto dict2 = dict;
  CHECK(dict2.count(z) == 1);
  dict.Set(zz, x);
  CHECK(dict2.count(zz) == 0);
  CHECK(dict.count(zz) == 1);

  auto it = dict.find(zz);
  CHECK(it != dict.end() && (*it).second.same_as(x));

  it = dict2.find(zz);
  CHECK(it == dict2.end());
}

TEST(Map, Iterator) {
  using namespace tvm;
  PrimExpr a = 1, b = 2;
  Map<PrimExpr, PrimExpr> map1{{a, b}};
  std::unordered_map<PrimExpr, PrimExpr, ObjectPtrHash, ObjectPtrEqual> map2(map1.begin(),
                                                                             map1.end());
  CHECK(map2[a].as<IntImmNode>()->value == 2);
}

TEST(Map, Insert) {
  using namespace tvm;
  auto check = [](const Map<String, Integer>& result,
                  std::unordered_map<std::string, int64_t> expected) {
    CHECK_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      CHECK(expected.count(kv.first));
      CHECK_EQ(expected[kv.first], kv.second.operator int64_t());
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
    CHECK_EQ(result.size(), expected.size());
    for (const auto& kv : result) {
      CHECK(expected.count(kv.first));
      CHECK_EQ(expected[kv.first], kv.second.operator int64_t());
      expected.erase(kv.first);
    }
  };
  Map<String, Integer> map{{"a", 1}, {"b", 2}, {"c", 3}, {"d", 4}, {"e", 5}};
  std::unordered_map<std::string, int64_t> stl(map.begin(), map.end());
  for (char c = 'a'; c <= 'e'; ++c) {
    Map<String, Integer> result = map;
    std::unordered_map<std::string, int64_t> expected(stl);
    std::string key(1, c);
    result.erase(key);
    expected.erase(key);
    check(result, expected);
  }
}

TEST(String, MoveFromStd) {
  using namespace std;
  string source = "this is a string";
  string expect = source;
  String s(std::move(source));
  string copy = (string)s;
  CHECK_EQ(copy, expect);
  CHECK_EQ(source.size(), 0);
}

TEST(String, CopyFromStd) {
  using namespace std;
  string source = "this is a string";
  string expect = source;
  String s{source};
  string copy = (string)s;
  CHECK_EQ(copy, expect);
  CHECK_EQ(source.size(), expect.size());
}

TEST(String, Assignment) {
  using namespace std;
  String s{string{"hello"}};
  s = string{"world"};
  CHECK_EQ(s == "world", true);
  string s2{"world2"};
  s = std::move(s2);
  CHECK_EQ(s == "world2", true);
}

TEST(String, empty) {
  using namespace std;
  String s{"hello"};
  CHECK_EQ(s.empty(), false);
  s = std::string("");
  CHECK_EQ(s.empty(), true);
}

TEST(String, Comparisons) {
  using namespace std;
  string source = "a string";
  string mismatch = "a string but longer";
  String s{source};
  String m{mismatch};

  CHECK_EQ(s == source, true);
  CHECK_EQ(s == mismatch, false);
  CHECK_EQ(s == source.data(), true);
  CHECK_EQ(s == mismatch.data(), false);

  CHECK_EQ(s < m, source < mismatch);
  CHECK_EQ(s > m, source > mismatch);
  CHECK_EQ(s <= m, source <= mismatch);
  CHECK_EQ(s >= m, source >= mismatch);
  CHECK_EQ(s == m, source == mismatch);
  CHECK_EQ(s != m, source != mismatch);

  CHECK_EQ(m < s, mismatch < source);
  CHECK_EQ(m > s, mismatch > source);
  CHECK_EQ(m <= s, mismatch <= source);
  CHECK_EQ(m >= s, mismatch >= source);
  CHECK_EQ(m == s, mismatch == source);
  CHECK_EQ(m != s, mismatch != source);
}

// Check '\0' handling
TEST(String, null_byte_handling) {
  using namespace std;
  // Ensure string still compares equal if it contains '\0'.
  string v1 = "hello world";
  size_t v1_size = v1.size();
  v1[5] = '\0';
  CHECK_EQ(v1[5], '\0');
  CHECK_EQ(v1.size(), v1_size);
  String str_v1{v1};
  CHECK_EQ(str_v1.compare(v1), 0);
  CHECK_EQ(str_v1.size(), v1_size);

  // Ensure bytes after '\0' are taken into account for mismatches.
  string v2 = "aaa one";
  string v3 = "aaa two";
  v2[3] = '\0';
  v3[3] = '\0';
  String str_v2{v2};
  String str_v3{v3};
  CHECK_EQ(str_v2.compare(str_v3), -1);
  CHECK_EQ(str_v2.size(), 7);
  // strcmp won't be able to detect the mismatch
  CHECK_EQ(strcmp(v2.data(), v3.data()), 0);
  // string::compare can handle \0 since it knows size
  CHECK_LT(v2.compare(v3), 0);

  // If there is mismatch before '\0', should still handle it.
  string v4 = "acc one";
  string v5 = "abb two";
  v4[3] = '\0';
  v5[3] = '\0';
  String str_v4{v4};
  String str_v5{v5};
  CHECK_GT(str_v4.compare(str_v5), 0);
  CHECK_EQ(str_v4.size(), 7);
  // strcmp is able to detect the mismatch
  CHECK_GT(strcmp(v4.data(), v5.data()), 0);
  // string::compare can handle \0 since it knows size
  CHECK_GT(v4.compare(v5), 0);
}

TEST(String, compare_same_memory_region_different_size) {
  using namespace std;
  string source = "a string";
  String str_source{source};
  char* memory = const_cast<char*>(str_source.data());
  CHECK_EQ(str_source.compare(memory), 0);
  // This changes the string size
  memory[2] = '\0';
  // memory is logically shorter now
  CHECK_GT(str_source.compare(memory), 0);
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
  CHECK_EQ(str_source.compare(source), 0);
  CHECK(str_source == source);
  CHECK(source == str_source);
  CHECK(str_source <= source);
  CHECK(source <= str_source);
  CHECK(str_source >= source);
  CHECK(source >= str_source);
  CHECK_LT(str_source.compare(mismatch1), 0);
  CHECK(str_source < mismatch1);
  CHECK(mismatch1 != str_source);
  CHECK_GT(str_source.compare(mismatch2), 0);
  CHECK(str_source > mismatch2);
  CHECK(mismatch2 < str_source);
  CHECK_GT(str_source.compare(mismatch3), 0);
  CHECK(str_source > mismatch3);
  CHECK_LT(str_source.compare(mismatch4), 0);
  CHECK(str_source < mismatch4);
  CHECK(mismatch4 > str_source);

  // compare with char*
  CHECK_EQ(str_source.compare(source.data()), 0);
  CHECK(str_source == source.data());
  CHECK(source.data() == str_source);
  CHECK(str_source <= source.data());
  CHECK(source <= str_source.data());
  CHECK(str_source >= source.data());
  CHECK(source >= str_source.data());
  CHECK_LT(str_source.compare(mismatch1.data()), 0);
  CHECK(str_source < mismatch1.data());
  CHECK(str_source != mismatch1.data());
  CHECK(mismatch1.data() != str_source);
  CHECK_GT(str_source.compare(mismatch2.data()), 0);
  CHECK(str_source > mismatch2.data());
  CHECK(mismatch2.data() < str_source);
  CHECK_GT(str_source.compare(mismatch3.data()), 0);
  CHECK(str_source > mismatch3.data());
  CHECK_LT(str_source.compare(mismatch4.data()), 0);
  CHECK(str_source < mismatch4.data());
  CHECK(mismatch4.data() > str_source);

  // compare with String
  CHECK_LT(str_source.compare(str_mismatch1), 0);
  CHECK(str_source < str_mismatch1);
  CHECK_GT(str_source.compare(str_mismatch2), 0);
  CHECK(str_source > str_mismatch2);
  CHECK_GT(str_source.compare(str_mismatch3), 0);
  CHECK(str_source > str_mismatch3);
  CHECK_LT(str_source.compare(str_mismatch4), 0);
  CHECK(str_source < str_mismatch4);
}

TEST(String, c_str) {
  using namespace std;
  string source = "this is a string";
  string mismatch = "mismatch";
  String s{source};

  CHECK_EQ(std::strcmp(s.c_str(), source.data()), 0);
  CHECK_NE(std::strcmp(s.c_str(), mismatch.data()), 0);
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

  CHECK_EQ(map[k1], v1);
  CHECK_EQ(map[k2], v2);
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

  CHECK_EQ(res1.compare("helloworld"), 0);
  CHECK_EQ(res2.compare("helloworld"), 0);
  CHECK_EQ(res3.compare("worldhello"), 0);
  CHECK_EQ(res4.compare("helloworld"), 0);
  CHECK_EQ(res5.compare("worldhello"), 0);
}

TEST(Optional, Composition) {
  Optional<String> opt0(nullptr);
  Optional<String> opt1 = String("xyz");
  Optional<String> opt2 = String("xyz1");
  // operator bool
  CHECK(!opt0);
  CHECK(opt1);
  // comparison op
  CHECK(opt0 != "xyz");
  CHECK(opt1 == "xyz");
  CHECK(opt1 != nullptr);
  CHECK(opt0 == nullptr);
  CHECK(opt0.value_or("abc") == "abc");
  CHECK(opt1.value_or("abc") == "xyz");
  CHECK(opt0 != opt1);
  CHECK(opt1 == Optional<String>(String("xyz")));
  CHECK(opt0 == Optional<String>(nullptr));
  opt0 = opt1;
  CHECK(opt0 == opt1);
  CHECK(opt0.value().same_as(opt1.value()));
  opt0 = std::move(opt2);
  CHECK(opt0 != opt2);
}

TEST(Optional, IntCmp) {
  Integer val(CallingConv::kDefault);
  Optional<Integer> opt = Integer(0);
  CHECK(0 == static_cast<int>(CallingConv::kDefault));
  CHECK(val == CallingConv::kDefault);
  CHECK(opt == CallingConv::kDefault);

  // check we can handle implicit 0 to nullptr conversion.
  Optional<Integer> opt1(nullptr);
  CHECK(opt1 != 0);
  CHECK(opt1 != false);
  CHECK(!(opt1 == 0));
}

TEST(Optional, PackedCall) {
  auto tf = [](Optional<String> s, bool isnull) {
    if (isnull) {
      CHECK(s == nullptr);
    } else {
      CHECK(s != nullptr);
    }
    return s;
  };
  auto func = TypedPackedFunc<Optional<String>(Optional<String>, bool)>(tf);
  CHECK(func(String("xyz"), false) == "xyz");
  CHECK(func(Optional<String>(nullptr), true) == nullptr);

  auto pf = [](TVMArgs args, TVMRetValue* rv) {
    Optional<String> s = args[0];
    bool isnull = args[1];
    if (isnull) {
      CHECK(s == nullptr);
    } else {
      CHECK(s != nullptr);
    }
    *rv = s;
  };
  auto packedfunc = PackedFunc(pf);
  CHECK(packedfunc("xyz", false).operator String() == "xyz");
  CHECK(packedfunc("xyz", false).operator Optional<String>() == "xyz");
  CHECK(packedfunc(nullptr, true).operator Optional<String>() == nullptr);

  // test FFI convention.
  auto test_ffi = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
    int tcode = args[1];
    CHECK_EQ(args[0].type_code(), tcode);
  });
  String s = "xyz";
  auto nd = NDArray::Empty({0, 1}, DataType::Float(32), DLContext{kDLCPU, 0});
  test_ffi(Optional<NDArray>(nd), static_cast<int>(kTVMNDArrayHandle));
  test_ffi(Optional<String>(s), static_cast<int>(kTVMObjectRValueRefArg));
  test_ffi(s, static_cast<int>(kTVMObjectHandle));
  test_ffi(String(s), static_cast<int>(kTVMObjectRValueRefArg));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
