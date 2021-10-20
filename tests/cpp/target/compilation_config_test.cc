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
#include <tvm/target/compilation_config.h>
#include <tvm/target/target.h>

namespace tvm {
namespace {

TEST(CompilationConfig, Constructor) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kDLCUDA)));

  Target cuda_target = Target("nvidia/tesla-p40");
  Target default_cpu_target = Target("llvm");

  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), cuda_target);

  CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target=*/{});

  ASSERT_EQ(config->legacy_target_map.size(), 1);
  EXPECT_EQ((*config->legacy_target_map.begin()).second->str(), cuda_target->str());
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_EQ(config->host_target->str(), default_cpu_target->str());
  ASSERT_EQ(config->primitive_targets.size(), 1);
  EXPECT_EQ(config->primitive_targets[0]->str(), cuda_target->str());
  EXPECT_EQ(config->default_primitive_se_scope->device_type(), kDLCUDA);
  EXPECT_EQ(config->default_primitive_se_scope->target()->str(), cuda_target->str());
  EXPECT_EQ(config->host_se_scope->device_type(), kDLCPU);
  EXPECT_EQ(config->host_se_scope->target()->str(), default_cpu_target->str());
  ASSERT_TRUE(config->optional_homogeneous_target.defined());
  EXPECT_EQ(config->optional_homogeneous_target->str(), cuda_target->str());
}

TEST(CompilationConfig, CanonicalSEScope) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  TargetMap legacy_target_map;
  Target cuda_target = Target("cuda");
  Target cpu_target = Target("llvm -mcpu arm64");
  Target host_target = Target("llvm");
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), cuda_target);
  legacy_target_map.Set(Integer(static_cast<int>(kDLCPU)), cpu_target);
  CompilationConfig config(pass_ctx, legacy_target_map, host_target);

  {
    SEScope in = SEScope(kDLCPU);
    SEScope actual = config->CanonicalSEScope(in);
    ASSERT_TRUE(actual->target().defined());
    EXPECT_EQ(actual->target()->str(), cpu_target->str());
    EXPECT_EQ(config->CanonicalSEScope(in), actual);
  }
  {
    SEScope in = SEScope(kDLCUDA);
    SEScope actual = config->CanonicalSEScope(in);
    ASSERT_TRUE(actual->target().defined());
    EXPECT_EQ(actual->target()->str(), cuda_target->str());
    EXPECT_EQ(config->CanonicalSEScope(in), actual);
  }
}

}  // namespace
}  // namespace tvm
