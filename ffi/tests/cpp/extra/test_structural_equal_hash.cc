
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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;
namespace refl = tvm::ffi::reflection;

TEST(StructuralEqualHash, Array) {
  Array<int> a = {1, 2, 3};
  Array<int> b = {1, 2, 3};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Array<int> c = {1, 3};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));
  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);

  // first directly interepret diff,
  EXPECT_TRUE(diff_a_c.has_value());
  auto lhs_steps = (*diff_a_c).get<0>()->ToSteps();
  auto rhs_steps = (*diff_a_c).get<1>()->ToSteps();
  EXPECT_EQ(lhs_steps[0]->kind, refl::AccessKind::kArrayItem);
  EXPECT_EQ(rhs_steps[0]->kind, refl::AccessKind::kArrayItem);
  EXPECT_EQ(lhs_steps[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ(rhs_steps[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ(lhs_steps.size(), 1);
  EXPECT_EQ(rhs_steps.size(), 1);

  // use structural equal for checking in future parts
  // given we have done some basic checks above by directly interepret diff,
  Array<int> d = {1, 2};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::FromSteps({
                                                    refl::AccessStep::ArrayItem(2),
                                                }),
                                                refl::AccessPath::FromSteps({
                                                    refl::AccessStep::ArrayItemMissing(2),
                                                }));
  // then use structural equal to check it
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));
}

TEST(StructuralEqualHash, Map) {
  // same map but different insertion order
  Map<String, int> a = {{"a", 1}, {"b", 2}, {"c", 3}};
  Map<String, int> b = {{"b", 2}, {"c", 3}, {"a", 1}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Map<String, int> c = {{"a", 1}, {"b", 2}, {"c", 4}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("c"),
                                                refl::AccessPath::Root()->MapItem("c"));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));
}

TEST(StructuralEqualHash, NestedMapArray) {
  Map<String, Array<Any>> a = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  Map<String, Array<Any>> b = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Map<String, Array<Any>> c = {{"a", {1, 2, 3}}, {"b", {4, "world", 6}}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c =
      refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b")->ArrayItem(1),
                           refl::AccessPath::Root()->MapItem("b")->ArrayItem(1));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));

  Map<String, Array<Any>> d = {{"a", {1, 2, 3}}};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b"),
                                                refl::AccessPath::Root()->MapItemMissing("b"));
  EXPECT_TRUE(diff_a_d.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));

  auto diff_d_a = StructuralEqual::GetFirstMismatch(d, a);
  auto expected_diff_d_a = refl::AccessPathPair(refl::AccessPath::Root()->MapItemMissing("b"),
                                                refl::AccessPath::Root()->MapItem("b"));
}

TEST(StructuralEqualHash, FreeVar) {
  TVar a = TVar("a");
  TVar b = TVar("b");
  EXPECT_TRUE(StructuralEqual::Equal(a, b, /*map_free_vars=*/true));
  EXPECT_FALSE(StructuralEqual::Equal(a, b));

  EXPECT_NE(StructuralHash()(a), StructuralHash()(b));
  EXPECT_EQ(StructuralHash::Hash(a, /*map_free_vars=*/true),
            StructuralHash::Hash(b, /*map_free_vars=*/true));
}

TEST(StructuralEqualHash, FuncDefAndIgnoreField) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TFunc fa = TFunc({x}, {TInt(1), x}, String("comment a"));
  TFunc fb = TFunc({y}, {TInt(1), y}, String("comment b"));

  TFunc fc = TFunc({x}, {TInt(1), TInt(2)}, String("comment c"));

  EXPECT_TRUE(StructuralEqual()(fa, fb));
  EXPECT_EQ(StructuralHash()(fa), StructuralHash()(fb));

  EXPECT_FALSE(StructuralEqual()(fa, fc));
  auto diff_fa_fc = StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc = refl::AccessPathPair(refl::AccessPath::FromSteps({
                                                      refl::AccessStep::Attr("body"),
                                                      refl::AccessStep::ArrayItem(1),
                                                  }),
                                                  refl::AccessPath::FromSteps({
                                                      refl::AccessStep::Attr("body"),
                                                      refl::AccessStep::ArrayItem(1),
                                                  }));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

TEST(StructuralEqualHash, CustomTreeNode) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TCustomFunc fa = TCustomFunc({x}, {TInt(1), x}, "comment a");
  TCustomFunc fb = TCustomFunc({y}, {TInt(1), y}, "comment b");

  TCustomFunc fc = TCustomFunc({x}, {TInt(1), TInt(2)}, "comment c");

  EXPECT_TRUE(StructuralEqual()(fa, fb));
  EXPECT_EQ(StructuralHash()(fa), StructuralHash()(fb));

  EXPECT_FALSE(StructuralEqual()(fa, fc));
  auto diff_fa_fc = StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc =
      refl::AccessPathPair(refl::AccessPath::Root()->Attr("body")->ArrayItem(1),
                           refl::AccessPath::Root()->Attr("body")->ArrayItem(1));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

}  // namespace
