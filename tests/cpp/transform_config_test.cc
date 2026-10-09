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
#include <tvm/ir/transform.h>

namespace tvm {
namespace transform {
namespace {

class TestConfigNode : public ffi::Object {
 public:
  static inline int constructions = 0;
  int limit;

  TestConfigNode() { ++constructions; }

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<TestConfigNode>().def_ro("limit", &TestConfigNode::limit,
                                                        ffi::reflection::DefaultValue(17));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TransformConfig", TestConfigNode, ffi::Object);
};

class TestConfig : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TestConfig, ffi::ObjectRef, TestConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  TestConfigNode::RegisterReflection();
  PassContext::RegisterConfigOption<TestConfig>("testing.transform_config");
}

TEST(TransformConfig, LazyFreshReflectionDefaults) {
  EXPECT_EQ(PassContext::ListConfigs().at("testing.transform_config").at("type"),
            "testing.TransformConfig");
  auto ctx = PassContext::Create();
  int before = TestConfigNode::constructions;
  auto first = ctx->GetConfigOrDefault<TestConfig>("testing.transform_config");
  auto second = ctx->GetConfigOrDefault<TestConfig>("testing.transform_config");
  EXPECT_EQ(TestConfigNode::constructions, before + 2);
  EXPECT_EQ(first->limit, 17);
  EXPECT_EQ(second->limit, 17);
  EXPECT_FALSE(first.same_as(second));
  EXPECT_FALSE(ctx->GetConfig<TestConfig>("testing.transform_config").has_value());

  auto configured_node = ffi::make_object<TestConfigNode>();
  configured_node->limit = 42;
  auto configured = ffi::GetRef<TestConfig>(configured_node.get());
  ctx->config.Set("testing.transform_config", configured);
  before = TestConfigNode::constructions;
  auto found = ctx->GetConfigOrDefault<TestConfig>("testing.transform_config");
  EXPECT_TRUE(found.same_as(configured));
  EXPECT_EQ(found->limit, 42);
  EXPECT_EQ(TestConfigNode::constructions, before);
}

TEST(TransformConfig, OptionalValuesAndTypeErrors) {
  auto ctx = PassContext::Create();
  EXPECT_FALSE(ctx->GetConfig<bool>("flag").has_value());
  EXPECT_TRUE(ctx->GetConfig<bool>("flag").value_or(true));
  ctx->config.Set("flag", false);
  ASSERT_TRUE(ctx->GetConfig<bool>("flag").has_value());
  EXPECT_FALSE(ctx->GetConfig<bool>("flag").value_or(true));
  ctx->config.Set("flag", ffi::String("invalid"));
  EXPECT_THROW(ctx->GetConfig<bool>("flag"), ffi::Error);
  ctx->config.Set("testing.transform_config", ffi::String("invalid"));
  int before = TestConfigNode::constructions;
  EXPECT_THROW(ctx->GetConfigOrDefault<TestConfig>("testing.transform_config"), ffi::Error);
  EXPECT_EQ(TestConfigNode::constructions, before);
}

}  // namespace
}  // namespace transform
}  // namespace tvm
