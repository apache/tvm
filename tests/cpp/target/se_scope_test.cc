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
namespace target {
namespace {

TEST(SEScope, MemoizedConstructors) {
  Target target_a = Target("cuda");
  Target target_b = Target("llvm");
  SEScope se_scope_a = SEScope::MakeSEScope(kDLCUDA, 3, target_a, "local");
  SEScope se_scope_b = SEScope::MakeSEScope(kDLCPU, 1, target_b, "global");

  EXPECT_EQ(SEScope::MakeSEScope(kDLCUDA, 3, target_a, "local"), se_scope_a);
  EXPECT_EQ(SEScope::MakeSEScope(kDLCPU, 1, target_b, "global"), se_scope_b);
  EXPECT_NE(SEScope::MakeSEScope(kDLCUDA, 2, target_a, "local"), se_scope_a);
  EXPECT_NE(SEScope::MakeSEScope(kDLCPU, 3, target_b, "local"), se_scope_a);
  EXPECT_NE(SEScope::MakeSEScope(kDLCUDA, 3, target_a, "global"), se_scope_a);
}

TEST(SEScope, Join_Defined) {
  Target target_a = Target("cuda");
  SEScope lhs = SEScope::MakeSEScope(kDLCUDA, 3);
  SEScope rhs = SEScope::MakeSEScope(kDLCUDA, -1, target_a, "global");
  Optional<SEScope> actual = SEScope::Join(lhs, rhs);
  EXPECT_TRUE(actual.operator bool());
  SEScope expected = SEScope::MakeSEScope(kDLCUDA, 3, target_a, "global");
  EXPECT_EQ(actual.value(), expected);
}

TEST(SEScope, Join_Undefined) {
  SEScope lhs = SEScope::MakeSEScope(kDLCUDA, 3);
  SEScope rhs = SEScope::MakeSEScope(kDLCUDA, 4);
  Optional<SEScope> actual = SEScope::Join(lhs, rhs);
  EXPECT_FALSE(actual);
}

TEST(SEScope, Default) {
  Target target_a = Target("cuda");
  SEScope lhs = SEScope::MakeSEScope(kDLCUDA, -1, Target(), "global");
  SEScope rhs = SEScope::MakeSEScope(kDLCUDA, 3, target_a, "local");
  SEScope actual = SEScope::Default(lhs, rhs);
  SEScope expected = SEScope::MakeSEScope(kDLCUDA, 3, target_a, "global");
  EXPECT_EQ(actual, expected);
}

TEST(CompilationConfig, Constructor) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kDLCUDA)));

  Target cuda_target = Target("nvidia/tesla-p40");
  Target default_cpu_target = Target("llvm");

  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), cuda_target);

  CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/Target());

  EXPECT_EQ(config.legacy_target_map.size(), 1);
  EXPECT_EQ((*config.legacy_target_map.begin()).second->str(), cuda_target->str());
  EXPECT_FALSE(config.optional_host_target.defined());
  EXPECT_EQ(config.targets.size(), 2);
  EXPECT_EQ(config.targets[0]->str(), cuda_target->str());
  EXPECT_EQ(config.targets[1]->str(), default_cpu_target->str());
  EXPECT_EQ(config.default_primitive_se_scope->device_type(), kDLCUDA);
  EXPECT_EQ(config.default_primitive_se_scope->target()->str(), cuda_target->str());
  EXPECT_EQ(config.host_se_scope->device_type(), kDLCPU);
  EXPECT_EQ(config.host_se_scope->target()->str(), default_cpu_target->str());
  EXPECT_EQ(config.homogeneous_target->str(), cuda_target->str());
}

}  // namespace
}  // namespace target
}  // namespace tvm
