
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
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/reflection/structural_equal.h>
#include <tvm/ffi/reflection/structural_hash.h>
#include <tvm/ffi/string.h>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;
namespace refl = tvm::ffi::reflection;

TEST(StructuralEqualHash, Array) {
  Array<int> a = {1, 2, 3};
  Array<int> b = {1, 2, 3};
  EXPECT_TRUE(refl::StructuralEqual()(a, b));
  EXPECT_EQ(refl::StructuralHash()(a), refl::StructuralHash()(b));

  Array<int> c = {1, 3};
  EXPECT_FALSE(refl::StructuralEqual()(a, c));
  EXPECT_NE(refl::StructuralHash()(a), refl::StructuralHash()(c));
  auto diff_a_c = refl::StructuralEqual::GetFirstMismatch(a, c);

  // first directly interepret diff,
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_EQ((*diff_a_c).get<0>()[0]->kind, refl::AccessKind::kArrayIndex);
  EXPECT_EQ((*diff_a_c).get<1>()[0]->kind, refl::AccessKind::kArrayIndex);
  EXPECT_EQ((*diff_a_c).get<0>()[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ((*diff_a_c).get<1>()[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ((*diff_a_c).get<0>().size(), 1);
  EXPECT_EQ((*diff_a_c).get<1>().size(), 1);

  // use structural equal for checking in future parts
  // given we have done some basic checks above by directly interepret diff,
  Array<int> d = {1, 2};
  auto diff_a_d = refl::StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath({
                                                    refl::AccessStep::ArrayIndex(2),
                                                }),
                                                refl::AccessPath({
                                                    refl::AccessStep::ArrayIndexMissing(2),
                                                }));
  // then use structural equal to check it
  EXPECT_TRUE(refl::StructuralEqual()(diff_a_d, expected_diff_a_d));
}

TEST(StructuralEqualHash, Map) {
  // same map but different insertion order
  Map<String, int> a = {{"a", 1}, {"b", 2}, {"c", 3}};
  Map<String, int> b = {{"b", 2}, {"c", 3}, {"a", 1}};
  EXPECT_TRUE(refl::StructuralEqual()(a, b));
  EXPECT_EQ(refl::StructuralHash()(a), refl::StructuralHash()(b));

  Map<String, int> c = {{"a", 1}, {"b", 2}, {"c", 4}};
  EXPECT_FALSE(refl::StructuralEqual()(a, c));
  EXPECT_NE(refl::StructuralHash()(a), refl::StructuralHash()(c));

  auto diff_a_c = refl::StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c = refl::AccessPathPair(refl::AccessPath({
                                                    refl::AccessStep::MapKey("c"),
                                                }),
                                                refl::AccessPath({
                                                    refl::AccessStep::MapKey("c"),
                                                }));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(refl::StructuralEqual()(diff_a_c, expected_diff_a_c));
}

TEST(StructuralEqualHash, NestedMapArray) {
  Map<String, Array<Any>> a = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  Map<String, Array<Any>> b = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  EXPECT_TRUE(refl::StructuralEqual()(a, b));
  EXPECT_EQ(refl::StructuralHash()(a), refl::StructuralHash()(b));

  Map<String, Array<Any>> c = {{"a", {1, 2, 3}}, {"b", {4, "world", 6}}};
  EXPECT_FALSE(refl::StructuralEqual()(a, c));
  EXPECT_NE(refl::StructuralHash()(a), refl::StructuralHash()(c));

  auto diff_a_c = refl::StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c = refl::AccessPathPair(refl::AccessPath({
                                                    refl::AccessStep::MapKey("b"),
                                                    refl::AccessStep::ArrayIndex(1),
                                                }),
                                                refl::AccessPath({
                                                    refl::AccessStep::MapKey("b"),
                                                    refl::AccessStep::ArrayIndex(1),
                                                }));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(refl::StructuralEqual()(diff_a_c, expected_diff_a_c));

  Map<String, Array<Any>> d = {{"a", {1, 2, 3}}};
  auto diff_a_d = refl::StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath({
                                                    refl::AccessStep::MapKey("b"),
                                                }),
                                                refl::AccessPath({
                                                    refl::AccessStep::MapKeyMissing("b"),
                                                }));
  EXPECT_TRUE(diff_a_d.has_value());
  EXPECT_TRUE(refl::StructuralEqual()(diff_a_d, expected_diff_a_d));

  auto diff_d_a = refl::StructuralEqual::GetFirstMismatch(d, a);
  auto expected_diff_d_a = refl::AccessPathPair(refl::AccessPath({
                                                    refl::AccessStep::MapKeyMissing("b"),
                                                }),
                                                refl::AccessPath({
                                                    refl::AccessStep::MapKey("b"),
                                                }));
}

TEST(StructuralEqualHash, FreeVar) {
  TVar a = TVar("a");
  TVar b = TVar("b");
  EXPECT_TRUE(refl::StructuralEqual::Equal(a, b, /*map_free_vars=*/true));
  EXPECT_FALSE(refl::StructuralEqual::Equal(a, b));

  EXPECT_NE(refl::StructuralHash()(a), refl::StructuralHash()(b));
  EXPECT_EQ(refl::StructuralHash::Hash(a, /*map_free_vars=*/true),
            refl::StructuralHash::Hash(b, /*map_free_vars=*/true));
}

TEST(StructuralEqualHash, FuncDefAndIgnoreField) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TFunc fa = TFunc({x}, {TInt(1), x}, "comment a");
  TFunc fb = TFunc({y}, {TInt(1), y}, "comment b");

  TFunc fc = TFunc({x}, {TInt(1), TInt(2)}, "comment c");

  EXPECT_TRUE(refl::StructuralEqual()(fa, fb));
  EXPECT_EQ(refl::StructuralHash()(fa), refl::StructuralHash()(fb));

  EXPECT_FALSE(refl::StructuralEqual()(fa, fc));
  auto diff_fa_fc = refl::StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc = refl::AccessPathPair(refl::AccessPath({
                                                      refl::AccessStep::ObjectField("body"),
                                                      refl::AccessStep::ArrayIndex(1),
                                                  }),
                                                  refl::AccessPath({
                                                      refl::AccessStep::ObjectField("body"),
                                                      refl::AccessStep::ArrayIndex(1),
                                                  }));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(refl::StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

TEST(StructuralEqualHash, CustomTreeNode) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TCustomFunc fa = TCustomFunc({x}, {TInt(1), x}, "comment a");
  TCustomFunc fb = TCustomFunc({y}, {TInt(1), y}, "comment b");

  TCustomFunc fc = TCustomFunc({x}, {TInt(1), TInt(2)}, "comment c");

  EXPECT_TRUE(refl::StructuralEqual()(fa, fb));
  EXPECT_EQ(refl::StructuralHash()(fa), refl::StructuralHash()(fb));

  EXPECT_FALSE(refl::StructuralEqual()(fa, fc));
  auto diff_fa_fc = refl::StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc = refl::AccessPathPair(refl::AccessPath({
                                                      refl::AccessStep::ObjectField("body"),
                                                      refl::AccessStep::ArrayIndex(1),
                                                  }),
                                                  refl::AccessPath({
                                                      refl::AccessStep::ObjectField("body"),
                                                      refl::AccessStep::ArrayIndex(1),
                                                  }));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(refl::StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

}  // namespace
