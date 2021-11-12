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
#include <tvm/target/se_scope.h>
#include <tvm/target/target.h>

namespace tvm {
namespace {

TEST(SEScope, Join_Defined) {
  {
    Target target_a = Target("cuda");
    SEScope lhs = SEScope(kDLCUDA, 3);
    SEScope rhs = SEScope(kDLCUDA, -1, target_a, "global");
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    SEScope expected = SEScope(kDLCUDA, 3, target_a, "global");
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
  {
    Target target_a = Target("cuda");
    SEScope lhs = SEScope(kDLCUDA, -1, target_a, "global");
    SEScope rhs = SEScope(kDLCUDA, 3);
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    SEScope expected = SEScope(kDLCUDA, 3, target_a, "global");
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
  {
    Target target_a = Target("cuda");
    SEScope lhs = SEScope(kDLCUDA);
    SEScope rhs = SEScope(kDLCUDA, 2, target_a);
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    SEScope expected = SEScope(kDLCUDA, 2, target_a);
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
  {
    Target target_a = Target("cuda");
    SEScope lhs = SEScope();
    SEScope rhs = SEScope(kDLCUDA, 3, target_a, "global");
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    SEScope expected = rhs;
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
}

TEST(SEScope, Join_Undefined) {
  {
    SEScope lhs = SEScope(kDLCUDA);
    SEScope rhs = SEScope(kDLCPU);
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
  {
    SEScope lhs = SEScope(kDLCUDA, 3);
    SEScope rhs = SEScope(kDLCUDA, 4);
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
  {
    SEScope lhs = SEScope(kDLCUDA, 3, Target("cuda"));
    SEScope rhs = SEScope(kDLCUDA, 3, Target("cuda"));
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
  {
    SEScope lhs = SEScope(kDLCUDA, 3, Target("cuda"), "local");
    SEScope rhs = SEScope(kDLCUDA, 3, Target("cuda"), "global");
    Optional<SEScope> actual = SEScope::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
}

TEST(SEScope, Default) {
  Target target_a = Target("cuda");
  SEScope lhs = SEScope(kDLCUDA, -1, Target(), "global");
  SEScope rhs = SEScope(kDLCUDA, 3, target_a, "local");
  SEScope actual = SEScope::Default(lhs, rhs);
  SEScope expected = SEScope(kDLCUDA, 3, target_a, "global");
  EXPECT_TRUE(StructuralEqual()(actual, expected));
}

TEST(SEScope, Constructor_Invalid) { EXPECT_ANY_THROW(SEScope(kDLCPU, -1, Target("cuda"))); }

TEST(SEScopeCache, Memoized) {
  SEScopeCache cache;
  Target target_a = Target("cuda");
  Target target_b = Target("llvm");
  SEScope se_scope_a = cache.Make(kDLCUDA, 3, target_a, "local");
  SEScope se_scope_b = cache.Make(kDLCPU, 1, target_b, "global");

  EXPECT_EQ(cache.Make(kDLCUDA, 3, target_a, "local"), se_scope_a);
  EXPECT_EQ(cache.Make(kDLCPU, 1, target_b, "global"), se_scope_b);
  EXPECT_NE(cache.Make(kDLCUDA, 2, target_a, "local"), se_scope_a);
  EXPECT_NE(cache.Make(kDLCPU, 3, target_b, "local"), se_scope_a);
  EXPECT_NE(cache.Make(kDLCUDA, 3, target_a, "global"), se_scope_a);
}

}  // namespace
}  // namespace tvm
