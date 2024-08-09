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

#include "../../../src/target/llvm/llvm_instance.h"

using namespace tvm;

TVM_REGISTER_TARGET_KIND("TestTargetKind", kDLCPU)
    .set_attr<std::string>("Attr1", "Value1")
    .add_attr_option<runtime::Bool>("my_bool")
    .add_attr_option<Array<String>>("your_names")
    .add_attr_option<Map<String, runtime::Int>>("her_maps");

TargetJSON TestTargetParser(TargetJSON target) {
  String mcpu = Downcast<String>(target.at("mcpu"));
  target.Set("mcpu", String("super_") + mcpu);
  target.Set("keys", Array<String>({"super"}));
  target.Set("features", Map<String, ObjectRef>{{"test", runtime::Bool(true)}});
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
      {"my_bool", runtime::Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {
          "her_maps",
          Map<String, runtime::Int>{
              {"a", runtime::Int(1)},
              {"b", runtime::Int(2)},
          },
      },
  };
  Target target = Target(config);
  ICHECK_EQ(target->kind, TargetKind::Get("TestTargetKind").value());
  ICHECK_EQ(target->tag, "");
  ICHECK(target->keys.empty());
  runtime::Bool my_bool = target->GetAttr<runtime::Bool>("my_bool").value();
  ICHECK_EQ(my_bool.operator bool(), true);
  Array<String> your_names = target->GetAttr<Array<String>>("your_names").value();
  ICHECK_EQ(your_names.size(), 2U);
  ICHECK_EQ(your_names[0], "junru");
  ICHECK_EQ(your_names[1], "jian");
  Map<String, runtime::Int> her_maps =
      target->GetAttr<Map<String, runtime::Int>>("her_maps").value();
  ICHECK_EQ(her_maps.size(), 2U);
  ICHECK_EQ(her_maps["a"], 1);
  ICHECK_EQ(her_maps["b"], 2);
}

TEST(TargetCreationFail, UnrecognizedConfigOption) {
  Map<String, ObjectRef> config = {
      {"my_bool", runtime::Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {"bad", ObjectRef(nullptr)},
      {
          "her_maps",
          Map<String, runtime::Int>{
              {"a", runtime::Int(1)},
              {"b", runtime::Int(2)},
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
          Map<String, runtime::Int>{
              {"a", runtime::Int(1)},
              {"b", runtime::Int(2)},
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
      {"my_bool", runtime::Bool("true")},
      {"your_names", Array<String>{"junru", "jian"}},
      {
          "her_maps",
          Map<String, runtime::Int>{
              {"a", runtime::Int(1)},
              {"b", runtime::Int(2)},
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
  ASSERT_EQ(test_target->keys.size(), 1);
  ASSERT_EQ(test_target->keys[0], "super");
}

TEST(TargetCreation, TargetFeatures) {
  Target test_target_with_parser("TestTargetParser -mcpu=woof");
  ASSERT_EQ(test_target_with_parser->GetFeature<runtime::Bool>("test").value(), true);

  Target test_target_no_parser("TestTargetKind");
  ASSERT_EQ(test_target_no_parser->GetFeature<runtime::Bool>("test"), nullptr);
  ASSERT_EQ(test_target_no_parser->GetFeature<runtime::Bool>("test", runtime::Bool(true)).value(),
            true);
}

TEST(TargetCreation, TargetFeaturesBeforeParser) {
  Map<String, ObjectRef> features = {{"test", runtime::Bool(true)}};
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

TVM_REGISTER_TARGET_KIND("TestStringKind", kDLCPU)
    .add_attr_option<String>("single")
    .add_attr_option<Array<String>>("array")
    .add_attr_option<Array<Array<String>>>("nested-array")
    .add_attr_option<Array<Array<Array<String>>>>("nested2-array");

TEST(TargetCreation, ProcessStrings) {
  Target test_target1("TestStringKind -single='\\'string with single quote'");
  ASSERT_TRUE(test_target1->GetAttr<String>("single"));
  String string1 = test_target1->GetAttr<String>("single").value();
  ASSERT_EQ(string1, "'string with single quote");

  Target test_target2("TestStringKind -single='\\\'\\\\\\'blah\\\\\\'\\\''");
  ASSERT_TRUE(test_target2->GetAttr<String>("single"));
  String string2 = test_target2->GetAttr<String>("single").value();
  ASSERT_EQ(string2, "'\\\'blah\\\''");

  Target test_target3("TestStringKind -array=-danny,-sammy=1,-kirby='string with space'");
  ASSERT_TRUE(test_target3->GetAttr<Array<String>>("array"));
  Array<String> array3 = test_target3->GetAttr<Array<String>>("array").value();
  ASSERT_EQ(array3[0], "-danny");
  ASSERT_EQ(array3[1], "-sammy=1");
  ASSERT_EQ(array3[2], "-kirby='string with space'");

  Target test_target4("TestStringKind -array='fred, foo, bar',baz");
  ASSERT_TRUE(test_target4->GetAttr<Array<String>>("array"));
  Array<String> array4 = test_target4->GetAttr<Array<String>>("array").value();
  ASSERT_EQ(array4[0], "fred, foo, bar");
  ASSERT_EQ(array4[1], "baz");

  Target test_target5("TestStringKind -array='fr\\'ed','f\\'oo',' bar,baz '");
  ASSERT_TRUE(test_target5->GetAttr<Array<String>>("array"));
  Array<String> array5 = test_target5->GetAttr<Array<String>>("array").value();
  ASSERT_EQ(array5[0], "fr'ed");
  ASSERT_EQ(array5[1], "f'oo");
  ASSERT_EQ(array5[2], "bar,baz");

  Target test_target6("TestStringKind -nested-array='foo0,foo1,foo2','bar0,bar1,bar2','baz0,baz1'");
  ASSERT_TRUE(test_target6->GetAttr<Array<Array<String>>>("nested-array"));
  Array<Array<String>> array6 = test_target6->GetAttr<Array<Array<String>>>("nested-array").value();
  ASSERT_EQ(array6[0][0], "foo0");
  ASSERT_EQ(array6[0][1], "foo1");
  ASSERT_EQ(array6[0][2], "foo2");
  ASSERT_EQ(array6[1][0], "bar0");
  ASSERT_EQ(array6[1][1], "bar1");
  ASSERT_EQ(array6[1][2], "bar2");
  ASSERT_EQ(array6[2][0], "baz0");
  ASSERT_EQ(array6[2][1], "baz1");

  Target test_target7(
      "TestStringKind -nested2-array="
      "'\\'foo0,foo1\\',\\'bar0,bar1\\',\\'baz0,baz1\\'',"
      "'\\'zing0,zing1\\',\\'fred\\''");

  ASSERT_TRUE(test_target7->GetAttr<Array<Array<Array<String>>>>("nested2-array"));
  Array<Array<Array<String>>> array7 =
      test_target7->GetAttr<Array<Array<Array<String>>>>("nested2-array").value();
  // {
  //   {foo0, foo1},
  //   {bar0, bar1},
  //   {baz0, baz1},
  // },
  // {
  //   {zing0, zing1},
  //   {fred},
  // }
  ASSERT_EQ(array7.size(), 2);
  ASSERT_EQ(array7[0].size(), 3);
  ASSERT_EQ(array7[0][0].size(), 2);
  ASSERT_EQ(array7[0][1].size(), 2);
  ASSERT_EQ(array7[0][2].size(), 2);
  ASSERT_EQ(array7[1].size(), 2);
  ASSERT_EQ(array7[1][0].size(), 2);
  ASSERT_EQ(array7[1][1].size(), 1);

  ASSERT_EQ(array7[0][0][0], "foo0");
  ASSERT_EQ(array7[0][0][1], "foo1");
  ASSERT_EQ(array7[0][1][0], "bar0");
  ASSERT_EQ(array7[0][1][1], "bar1");
  ASSERT_EQ(array7[0][2][0], "baz0");
  ASSERT_EQ(array7[0][2][1], "baz1");
  ASSERT_EQ(array7[1][0][0], "zing0");
  ASSERT_EQ(array7[1][0][1], "zing1");
  ASSERT_EQ(array7[1][1][0], "fred");
}

#ifdef TVM_LLVM_VERSION
// Checks that malformed options cause an assertion.
TEST(TargetCreation, LLVMCommandLineParseFatalDashDashDash) {
  tvm::codegen::LLVMInstance inst;

  // Too many dashes in an otherwise valid option.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='---unroll-factor:uint=0'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalColonNoType) {
  tvm::codegen::LLVMInstance inst;

  // : not followed by type.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalColonNoTypeEqNoValue) {
  tvm::codegen::LLVMInstance inst;

  // : and = without type/value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:='");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalColonTypeNoEqNoValue) {
  tvm::codegen::LLVMInstance inst;

  // Option with type, but no = and no value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:bool'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalColonTypeEqNoValue) {
  tvm::codegen::LLVMInstance inst;

  // Option with type and =, but no value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:bool='");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalInvalidType) {
  tvm::codegen::LLVMInstance inst;

  // Option with invalid type.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:invalidtype=xyz'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalInvalidValue1) {
  tvm::codegen::LLVMInstance inst;

  // (Implicit) bool option without type, but with invalid value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option=2'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalInvalidValue2) {
  tvm::codegen::LLVMInstance inst;

  // Bool option without type, but with invalid value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option=fred'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalInvalidValue3) {
  tvm::codegen::LLVMInstance inst;

  // Bool option with type and =, but invalid value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:bool=2'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalInvalidValue4) {
  tvm::codegen::LLVMInstance inst;

  // Int option with invalid value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:int=haha'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineParseFatalInvalidValue5) {
  tvm::codegen::LLVMInstance inst;

  // UInt option with invalid value.
  EXPECT_THROW(
      {
        Target test_target("llvm -cl-opt='-option:uint=haha'");
        tvm::codegen::LLVMTargetInfo info(inst, test_target);
      },
      std::exception);
}

TEST(TargetCreation, LLVMCommandLineError) {
  tvm::codegen::LLVMInstance inst;

  // Check that invalid LLVM options are ignored.
  Target test_target("llvm -cl-opt=-not-an-option:uint=123");
  tvm::codegen::LLVMTargetInfo info(inst, test_target);
  ASSERT_TRUE(info.GetCommandLineOptions().empty());
}

TEST(TargetCreation, LLVMCommandLineSaveRestore) {
  tvm::codegen::LLVMInstance inst;

  // Check detection of modified global state
  Target test_target("llvm -cl-opt=-print-after-all");  // "false" by default
  tvm::codegen::LLVMTargetInfo info(inst, test_target);
  ASSERT_FALSE(info.MatchesGlobalState());
  {
    // Check that we can modify global state.
    tvm::codegen::LLVMTarget llvm_target(inst, info);
    ASSERT_TRUE(info.MatchesGlobalState());
  }
  // Check that we restored global state.
  ASSERT_FALSE(info.MatchesGlobalState());
}

TEST(TargetCreation, DetectSystemTriple) {
  Map<String, ObjectRef> config = {
      {"kind", String("llvm")},
  };

  Target target = Target(config);
  ICHECK_EQ(target->kind, TargetKind::Get("llvm").value());

  auto pf = tvm::runtime::Registry::Get("target.llvm_get_system_triple");
  if (pf == nullptr) {
    GTEST_SKIP() << "LLVM is not available, skipping test";
  }

  Optional<String> mtriple = target->GetAttr<String>("mtriple");
  ASSERT_TRUE(mtriple.value() == String((*pf)()));
}

#endif

TVM_REGISTER_TARGET_KIND("test_external_codegen_0", kDLCUDA)
    .set_attr<runtime::Bool>(tvm::attr::kIsExternalCodegen, runtime::Bool(true));

TVM_REGISTER_TARGET_KIND("test_external_codegen_1", kDLCUDA)
    .set_attr<runtime::Bool>(tvm::attr::kIsExternalCodegen, runtime::Bool(true));

TVM_REGISTER_TARGET_KIND("test_external_codegen_2", kDLMetal)
    .set_attr<runtime::Bool>(tvm::attr::kIsExternalCodegen, runtime::Bool(true));

TVM_REGISTER_TARGET_KIND("test_external_codegen_3", kDLCPU)
    .set_attr<tvm::relay::transform::FTVMRelayToTIR>(tvm::attr::kRelayToTIR,
                                                     tvm::relay::transform::InferType());

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
}
