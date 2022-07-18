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
#include <tvm/ir/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/target/target.h>

#include <cmath>
#include <string>

using namespace tvm;

TVM_REGISTER_TARGET_KIND("TestTargetKind", kDLCPU)
    .set_attr<std::string>("Attr1", "Value1")
    .add_attr_option<Bool>("my_bool")
    .add_attr_option<Array<String>>("your_names")
    .add_attr_option<Map<String, Integer>>("her_maps");

TargetJSON TestTargetParser(TargetJSON target) {
  String mcpu = Downcast<String>(target.at("mcpu"));
  target.Set("mcpu", String("super_") + mcpu);
  target.Set("keys", Array<String>({"super"}));
  target.Set("features", Map<String, ObjectRef>{{"test", Bool(true)}});
  return target;
}

Map<String, ObjectRef> TestAttrsPreProcessor(Map<String, ObjectRef> attrs) {
  attrs.Set("mattr", String("woof"));
  return attrs;
}

TVM_REGISTER_TARGET_KIND("TestTargetParser", kDLCPU)
    .add_attr_option<String>("mattr")
    .add_attr_option<String>("mcpu")
    .set_default_keys({"cpu"})
    .set_target_parser(TestTargetParser);

TVM_REGISTER_TARGET_KIND("TestAttrsPreprocessor", kDLCPU)
    .add_attr_option<String>("mattr")
    .set_default_keys({"cpu"})
    .set_attrs_preprocessor(TestAttrsPreProcessor);

TVM_REGISTER_TARGET_KIND("TestClashingPreprocessor", kDLCPU)
    .add_attr_option<String>("mattr")
    .add_attr_option<String>("mcpu")
    .set_default_keys({"cpu"})
    .set_attrs_preprocessor(TestAttrsPreProcessor)
    .set_target_parser(TestTargetParser);

TEST(TargetKind, GetAttrMap) {
  auto map = tvm::TargetKind::GetAttrMap<std::string>("Attr1");
  auto target_kind = tvm::TargetKind::Get("TestTargetKind").value();
  std::string result = map[target_kind];
  ICHECK_EQ(result, "Value1");
}

TEST(TargetCreation, NestedConfig) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  Target target = Target(config);
  ICHECK_EQ(target->kind, TargetKind::Get("TestTargetKind").value());
  ICHECK_EQ(target->tag, "");
  ICHECK(target->keys.empty());
  Bool my_bool = target->GetAttr<Bool>("my_bool").value();
  ICHECK_EQ(my_bool.operator bool(), true);
  Array<String> your_names = target->GetAttr<Array<String>>("your_names").value();
  ICHECK_EQ(your_names.size(), 2U);
  ICHECK_EQ(your_names[0], "junru");
  ICHECK_EQ(your_names[1], "jian");
  Map<String, Integer> her_maps = target->GetAttr<Map<String, Integer>>("her_maps").value();
  ICHECK_EQ(her_maps.size(), 2U);
  ICHECK_EQ(her_maps["a"], 1);
  ICHECK_EQ(her_maps["b"], 2);
}

TEST(TargetCreationFail, UnrecognizedConfigOption) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {"bad", ObjectRef(nullptr)},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  bool failed = false;
  try {
    Target tgt(config);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TargetCreationFail, TypeMismatch) {
  Map<String, ObjectRef> config = {
      {"my_bool", String("true")},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  bool failed = false;
  try {
    Target tgt(config);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TargetCreationFail, TargetKindNotFound) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool("true")},
      {"your_names", Array<String>{"junru", "jian"}},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  bool failed = false;
  try {
    Target tgt(config);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TargetCreation, TargetParser) {
  Target test_target("TestTargetParser -mcpu=woof");
  ASSERT_EQ(test_target->GetAttr<String>("mcpu").value(), "super_woof");
  ASSERT_EQ(test_target->keys.size(), 2);
  ASSERT_EQ(test_target->keys[0], "super");
  ASSERT_EQ(test_target->keys[1], "cpu");
}

TEST(TargetCreation, TargetFeatures) {
  Target test_target_with_parser("TestTargetParser -mcpu=woof");
  ASSERT_EQ(test_target_with_parser->GetFeature<Bool>("test").value(), true);

  Target test_target_no_parser("TestTargetKind");
  ASSERT_EQ(test_target_no_parser->GetFeature<Bool>("test"), nullptr);
  ASSERT_EQ(test_target_no_parser->GetFeature<Bool>("test", Bool(true)).value(), true);
}

TEST(TargetCreation, TargetFeaturesBeforeParser) {
  Map<String, ObjectRef> features = {{"test", Bool(true)}};
  Map<String, ObjectRef> config = {
      {"kind", String("TestTargetParser")},
      {"mcpu", String("woof")},
      {"features", features},
  };
  EXPECT_THROW(Target test(config), InternalError);
}

TEST(TargetCreation, TargetAttrsPreProcessor) {
  Target test_target("TestAttrsPreprocessor -mattr=cake");
  ASSERT_EQ(test_target->GetAttr<String>("mattr").value(), "woof");
}

TEST(TargetCreation, ClashingTargetProcessing) {
  EXPECT_THROW(Target test("TestClashingPreprocessor -mcpu=woof -mattr=cake"), InternalError);
}

TVM_REGISTER_TARGET_KIND("test_external_codegen_0", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

TVM_REGISTER_TARGET_KIND("test_external_codegen_1", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

TVM_REGISTER_TARGET_KIND("test_external_codegen_2", kDLMetal)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

TVM_REGISTER_TARGET_KIND("test_external_codegen_3", kDLCPU)
    .set_attr<FTVMRelayToTIR>(tvm::attr::kRelayToTIR, tvm::relay::transform::InferType());

TEST(Target, ExternalCodegen) {
  Target regular("cuda");
  Target external0("test_external_codegen_0");
  Target external1("test_external_codegen_1");
  Target external2("test_external_codegen_2");
  Target external3("test_external_codegen_3");

  ASSERT_FALSE(regular.IsExternalCodegen());
  ASSERT_TRUE(external0.IsExternalCodegen());
  ASSERT_TRUE(external1.IsExternalCodegen());
  ASSERT_TRUE(external2.IsExternalCodegen());
  ASSERT_TRUE(external3.IsExternalCodegen());

  ASSERT_TRUE(external0.IsExternalCodegenFor(regular));
  ASSERT_FALSE(regular.IsExternalCodegenFor(external0));
  ASSERT_TRUE(external1.IsExternalCodegenFor(regular));
  ASSERT_FALSE(regular.IsExternalCodegenFor(external1));
  ASSERT_FALSE(external2.IsExternalCodegenFor(regular));
  ASSERT_FALSE(regular.IsExternalCodegenFor(external2));
}

TEST(TargetCreation, DeduplicateKeys) {
  Map<String, ObjectRef> config = {
      {"kind", String("llvm")},
      {"keys", Array<String>{"cpu", "arm_cpu"}},
      {"device", String("arm_cpu")},
  };
  Target target = Target(config);
  ICHECK_EQ(target->kind, TargetKind::Get("llvm").value());
  ICHECK_EQ(target->tag, "");
  ICHECK_EQ(target->keys.size(), 2U);
  ICHECK_EQ(target->keys[0], "cpu");
  ICHECK_EQ(target->keys[1], "arm_cpu");
  ICHECK_EQ(target->attrs.size(), 2U);
  ICHECK_EQ(target->GetAttr<String>("device"), "arm_cpu");
  ICHECK_EQ(target->GetAttr<Bool>("link-params"), false);
}

TEST(TargetKindRegistry, ListTargetKinds) {
  Array<String> names = TargetKindRegEntry::ListTargetKinds();
  ICHECK_EQ(names.empty(), false);
  ICHECK_EQ(std::count(std::begin(names), std::end(names), "llvm"), 1);
}

TEST(TargetKindRegistry, ListTargetOptions) {
  TargetKind llvm = TargetKind::Get("llvm").value();
  Map<String, String> attrs = TargetKindRegEntry::ListTargetKindOptions(llvm);
  ICHECK_EQ(attrs.empty(), false);

  ICHECK_EQ(attrs["mattr"], "Array");
  ICHECK_EQ(attrs["mcpu"], "runtime.String");
  ICHECK_EQ(attrs["system-lib"], "IntImm");
}
