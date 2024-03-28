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

#include "../src/target/parsers/aprofile.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "../src/target/llvm/llvm_instance.h"

namespace tvm {
namespace target {
namespace parsers {
namespace aprofile {

using ::testing::HasSubstr;

static float defaultI8MM = 8.6;
static float optionalI8MM[] = {8.2, 8.3, 8.4, 8.5};
static float defaultDotProd = 8.4;
static float optionalDotProd[] = {8.2, 8.3};
static float optionalSME[] = {9.2, 9.3};

static bool CheckArchitectureAvailability() {
#if TVM_LLVM_VERSION > 120
  auto llvm_instance = std::make_unique<codegen::LLVMInstance>();
  codegen::LLVMTargetInfo llvm_backend(*llvm_instance, "llvm");
  Array<String> targets = llvm_backend.GetAllLLVMTargets();
  int expected_target_count = 0;
  for (String target : targets) {
    if (target == "aarch64" || target == "arm") {
      expected_target_count += 1;
    }
  }
  if (expected_target_count >= 2) {
    return true;
  }
#endif
  return false;
}
static bool has_aarch64_and_arm_targets = CheckArchitectureAvailability();

class AProfileParser : public ::testing::Test {
 public:
  // Check that LLVM has been compiled with the required targets, otherwise skip the test.
  // Unfortunately, googletest doesn't let you call GTEST_SKIP in SetUpTestSuite() to skip
  // the whole suite of tests, so a cached result is checked before each test is run instead.
  void SetUp() override {
    if (!has_aarch64_and_arm_targets) {
      GTEST_SKIP() << "Skipping as LLVM has not been built for Arm(R)-based targets.";
    }
  }
};

class AProfileParserTestWithParam : public AProfileParser,
                                    public testing::WithParamInterface<float> {};

static TargetFeatures ParseTargetWithAttrs(String mcpu, String mtriple, Array<String> mattr) {
  TargetJSON target_json = {
      {"kind", String("llvm")},
      {"mtriple", mtriple},
      {"mattr", mattr},
  };
  if (mcpu != "") {
    target_json.Set("mcpu", mcpu);
  }
  return ParseTarget(target_json);
}

std::string FloatToStringWithoutTrailingZeros(float value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

TEST_F(AProfileParser, ParseTargetKeys) {
  TargetJSON target = ParseTarget({{"kind", String("llvm")}});
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "arm_cpu");
  ASSERT_EQ(keys[1], "cpu");
}

TEST_F(AProfileParser, ParseTargetWithExistingKeys) {
  TargetJSON target = ParseTarget({
      {"kind", String("llvm")},
      {"keys", Array<String>{"cpu"}},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST_F(AProfileParser, ParseTargetWithDuplicateKey) {
  TargetJSON target = ParseTarget({
      {"kind", String("llvm")},
      {"keys", Array<String>{"cpu", "arm_cpu"}},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST_F(AProfileParser, ParseTargetDefaults) {
  TargetJSON target = ParseTarget({{"kind", String("llvm")}});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));

  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);
}

TEST_F(AProfileParser, IsAArch64Triple) {
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), true);
}

TEST_F(AProfileParser, IsAArch32Triple) {
  TargetJSON target = ParseTargetWithAttrs("", "armv7a-arm-none-eabi", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);

  target = ParseTargetWithAttrs("", "armv8a-arm-none-eabi", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);

  target = ParseTargetWithAttrs("", "arm-unknown-linux-gnu", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);
}

TEST_F(AProfileParser, IsAArch32BlankCPU) {
  TargetJSON target = ParseTarget({
      {"kind", String("llvm")},
      {"mtriple", String("arm-unknown-linux-gnu")},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
}

TEST_F(AProfileParser, IsAArch32TripleWithAProfile) {
  TargetJSON target = ParseTargetWithAttrs("cortex-a53", "armv7a-arm-none-eabi", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);

  target = ParseTargetWithAttrs("cortex-a53", "armv8a-arm-none-eabi", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);

  target = ParseTargetWithAttrs("cortex-a53", "arm-unknown-linux-gnu", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);
}

TEST_F(AProfileParser, IsAArch32TripleWithMProfile) {
  TargetJSON target = ParseTargetWithAttrs("cortex-m33", "armv7a-arm-none-eabi", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), false);

  target = ParseTargetWithAttrs("cortex-m33", "armv8a-arm-none-eabi", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), false);

  target = ParseTargetWithAttrs("cortex-m33", "arm-unknown-linux-gnu", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), false);
}

TEST_F(AProfileParser, AArch64HasASIMD) {
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);
}

TEST_F(AProfileParser, AArch32ASIMD) {
  TargetJSON target = ParseTargetWithAttrs("", "armv8a-arm-none-eabi", {});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);
}

TEST_F(AProfileParser, AArch32HasASIMDWithOption) {
  TargetJSON target = ParseTargetWithAttrs("", "armv8a-arm-none-eabi", {"+simd"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);
}

TEST_F(AProfileParser, AArch32HasASIMDWithAlternativeOption) {
  TargetJSON target = ParseTargetWithAttrs("", "armv8a-arm-none-eabi", {"+neon"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);
}

TEST_F(AProfileParser, DefaultI8MMSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultI8MM) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), true);
}

TEST_F(AProfileParser, DefaultI8MMSupportDisable) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultI8MM) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "-i8mm"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), false);
}

using AProfileOptionalI8MM = AProfileParserTestWithParam;
TEST_P(AProfileOptionalI8MM, OptionalI8MMSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), false);

  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+i8mm"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), true);
}

TEST_F(AProfileParser, DefaultDotProdSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultDotProd) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), true);
}

TEST_F(AProfileParser, DefaultDotProdSupportDisable) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultDotProd) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "-dotprod"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);
}

using AProfileOptionalDotProd = AProfileParserTestWithParam;
TEST_P(AProfileOptionalDotProd, OptionalDotProdSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);

  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+dotprod"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), true);
}

TEST_F(AProfileParser, ArchVersionInvalidLetter) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultDotProd) + "b";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);
}

using AProfileOptionalSVE = AProfileParserTestWithParam;
TEST_P(AProfileOptionalSVE, OptionalSVESupport) {
  const std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  // Check that the "has_sve" feature is not set by default when "+sve" isn't set as an attribute.
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_FALSE(Downcast<Bool>(features.at("has_sve")));

  // Check that the "has_sve" feature is set when "+sve" is explicitly set as an attribute.
  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_sve")));
}

TEST_F(AProfileParser, DefaultSVESupportSVESupport) {
  const std::string arch_attr = "+v9a";

  // Check that the "has_sve" feature is not set by default when "+sve" isn't set as an attribute.
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_sve")));

  // Check that the "has_sve" feature is set when "+sve" is explicitly set as an attribute.
  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_sve")));
}

using AProfileOptionalFP16 = AProfileParserTestWithParam;
TEST_P(AProfileOptionalFP16, OptionalFP16Support) {
  const std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  // Check that the "has_fp16_simd" feature is not set by default when "+fullfp16" isn't set as an
  // attribute.
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_FALSE(Downcast<Bool>(features.at("has_fp16_simd")));

  // Check that the "has_fp16_simd" feature is set when "+fullfp16" is explicitly set as an
  // attribute.
  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+fullfp16"});
  features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_fp16_simd")));

  // Check that the "has_fp16_simd" feature is set when "+sve" is explicitly set as an attribute.
  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_fp16_simd")));
}

TEST_F(AProfileParser, DefaultFP16Support) {
  const std::string arch_attr = "+v9a";

  // Check that the "has_fp16_simd" feature is not set by default when "+fullfp16" isn't set as an
  // attribute.
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_fp16_simd")));

  // Check that the "has_fp16_simd" feature is set when "+fullfp16" is explicitly set as an
  // attribute.
  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+fullfp16"});
  features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_fp16_simd")));

  // Check that the "has_fp16_simd" feature is set when "+sve" is explicitly set as an attribute.
  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_fp16_simd")));
}

TEST_F(AProfileParser, ImpliedFeature) {
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-linux-gnu", {"+sve2"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_sve")));
  EXPECT_TRUE(Downcast<Bool>(features.at("has_asimd")));
}

TEST_F(AProfileParser, UnexpectedTargetKind) {
  EXPECT_THROW(
      {
        try {
          ParseTarget({{"kind", String("c")}});
        } catch (const tvm::InternalError& e) {
          EXPECT_THAT(e.what(), HasSubstr("Expected target kind 'llvm', but got 'c'"));
          throw;
        }
      },
      tvm::InternalError);
}

TEST(AProfileParserInvalid, LLVMUnsupportedArchitecture) {
  if (has_aarch64_and_arm_targets) {
    GTEST_SKIP() << "LLVM has been compiled for the correct targets.";
  }
  TargetJSON target = ParseTarget({{"kind", String("llvm")}});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  for (auto feature : features) {
    ASSERT_EQ(Downcast<Bool>(feature.second), false);
  }
}

using AProfileOptionalSME = AProfileParserTestWithParam;
TEST_P(AProfileOptionalSME, OptionalSMESupport) {
  const std::string arch_attr = "+v9a";

  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_TRUE(IsArch(target));
  ASSERT_FALSE(Downcast<Bool>(features.at("has_sme")));

  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sme"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_TRUE(IsArch(target));
  ASSERT_TRUE(Downcast<Bool>(features.at("has_sme")));
}

INSTANTIATE_TEST_SUITE_P(AProfileParser, AProfileOptionalI8MM, ::testing::ValuesIn(optionalI8MM));
INSTANTIATE_TEST_SUITE_P(AProfileParser, AProfileOptionalDotProd,
                         ::testing::ValuesIn(optionalDotProd));
INSTANTIATE_TEST_SUITE_P(AProfileParser, AProfileOptionalSVE,
                         ::testing::Values(8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9));
INSTANTIATE_TEST_SUITE_P(AProfileParser, AProfileOptionalFP16,
                         ::testing::Values(8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9));
INSTANTIATE_TEST_SUITE_P(AProfileParser, AProfileOptionalSME, ::testing::ValuesIn(optionalSME));

}  // namespace aprofile
}  // namespace parsers
}  // namespace target
}  // namespace tvm
