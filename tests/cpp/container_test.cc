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
#include <tvm/tir/op.h>
#include <tvm/runtime/container.h>
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
  TestErrorSwitch(const TestErrorSwitch& other)
      : should_fail(other.should_fail) {
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

class TestArrayObj : public Object,
                     public InplaceArrayBase<TestArrayObj, TestErrorSwitch> {
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
    auto ptr =
        make_inplace_array_object<TestArrayObj, TestErrorSwitch>(fields.size());
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
    auto ptr =
        make_inplace_array_object<TestArrayObj, TestErrorSwitch>(fields.size());
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

TEST(StrMap, Expr) {
  using namespace tvm;
  Var x("x");
  auto z = max(x + 1 + 2, 100);
  Map<std::string, PrimExpr> dict{{"x", z}, {"z", 2}};
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
  CHECK(it == dict.end());

  LOG(INFO) << dict;
}

TEST(Map, Iterator) {
  using namespace tvm;
  PrimExpr a = 1, b = 2;
  Map<PrimExpr, PrimExpr> map1{{a, b}};
  std::unordered_map<PrimExpr, PrimExpr, ObjectHash, ObjectEqual>
      map2(map1.begin(), map1.end());
  CHECK(map2[a].as<IntImmNode>()->value == 2);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
