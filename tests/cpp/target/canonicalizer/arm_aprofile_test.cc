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

#include "../src/target/canonicalizer/llvm/arm_aprofile.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "../src/target/llvm/llvm_instance.h"

namespace tvm {
namespace target {
namespace canonicalizer {
namespace llvm {
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
  ffi::Array<ffi::String> targets = llvm_backend.GetAllLLVMTargets();
  int expected_target_count = 0;
  for (ffi::String target : targets) {
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

class AProfileCanonicalizerTest : public ::testing::Test {
 public:
  void SetUp() override {
    if (!has_aarch64_and_arm_targets) {
      GTEST_SKIP() << "Skipping as LLVM has not been built for Arm(R)-based targets.";
    }
  }
};

class AProfileCanonicalizerTestWithParam : public AProfileCanonicalizerTest,
                                           public testing::WithParamInterface<float> {};

static ffi::Map<ffi::String, ffi::Any> CanonicalizeTargetWithAttrs(ffi::String mcpu,
                                                                   ffi::String mtriple,
                                                                   ffi::Array<ffi::String> mattr) {
  ffi::Map<ffi::String, ffi::Any> target_json = {
      {"kind", ffi::String("llvm")},
      {"mtriple", mtriple},
      {"mattr", mattr},
  };
  if (mcpu != "") {
    target_json.Set("mcpu", mcpu);
  }
  return Canonicalize(target_json);
}

std::string FloatToStringWithoutTrailingZeros(float value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

TEST_F(AProfileCanonicalizerTest, CanonicalizeTargetKeys) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({{"kind", ffi::String("llvm")}});
  ffi::Array<ffi::String> keys = Downcast<ffi::Array<ffi::String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "arm_cpu");
  ASSERT_EQ(keys[1], "cpu");
}

TEST_F(AProfileCanonicalizerTest, CanonicalizeTargetWithExistingKeys) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({
      {"kind", ffi::String("llvm")},
      {"keys", ffi::Array<ffi::String>{"cpu"}},
  });
  ffi::Array<ffi::String> keys = Downcast<ffi::Array<ffi::String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST_F(AProfileCanonicalizerTest, CanonicalizeTargetWithDuplicateKey) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({
      {"kind", ffi::String("llvm")},
      {"keys", ffi::Array<ffi::String>{"cpu", "arm_cpu"}},
  });
  ffi::Array<ffi::String> keys = Downcast<ffi::Array<ffi::String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST_F(AProfileCanonicalizerTest, CanonicalizeTargetDefaults) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({{"kind", ffi::String("llvm")}});

  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), false);
}

TEST_F(AProfileCanonicalizerTest, IsAArch64Triple) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), true);
}

TEST_F(AProfileCanonicalizerTest, IsAArch32Triple) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "armv7a-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), false);

  target = CanonicalizeTargetWithAttrs("", "armv8a-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), false);

  target = CanonicalizeTargetWithAttrs("", "arm-unknown-linux-gnu", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), false);
}

TEST_F(AProfileCanonicalizerTest, IsAArch32BlankCPU) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({
      {"kind", ffi::String("llvm")},
      {"mtriple", ffi::String("arm-unknown-linux-gnu")},
  });
  ASSERT_EQ(IsArch(target), true);
}

TEST_F(AProfileCanonicalizerTest, IsAArch32TripleWithAProfile) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("cortex-a53", "armv7a-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), false);

  target = CanonicalizeTargetWithAttrs("cortex-a53", "armv8a-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), false);

  target = CanonicalizeTargetWithAttrs("cortex-a53", "arm-unknown-linux-gnu", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.is_aarch64")), false);
}

TEST_F(AProfileCanonicalizerTest, IsAArch32TripleWithMProfile) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("cortex-m33", "armv7a-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), false);

  target = CanonicalizeTargetWithAttrs("cortex-m33", "armv8a-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), false);

  target = CanonicalizeTargetWithAttrs("cortex-m33", "arm-unknown-linux-gnu", {""});
  ASSERT_EQ(IsArch(target), false);
}

TEST_F(AProfileCanonicalizerTest, AArch64HasASIMD) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {""});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_asimd")), true);
}

TEST_F(AProfileCanonicalizerTest, AArch32ASIMD) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "armv8a-arm-none-eabi", {});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_asimd")), true);
}

TEST_F(AProfileCanonicalizerTest, AArch32HasASIMDWithOption) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "armv8a-arm-none-eabi", {"+simd"});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_asimd")), true);
}

TEST_F(AProfileCanonicalizerTest, AArch32HasASIMDWithAlternativeOption) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "armv8a-arm-none-eabi", {"+neon"});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_asimd")), true);
}

TEST_F(AProfileCanonicalizerTest, DefaultI8MMSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultI8MM) + "a";
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_matmul_i8")), true);
}

TEST_F(AProfileCanonicalizerTest, DefaultI8MMSupportDisable) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultI8MM) + "a";
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "-i8mm"});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_matmul_i8")), false);
}

using AProfileOptionalI8MM = AProfileCanonicalizerTestWithParam;
TEST_P(AProfileOptionalI8MM, OptionalI8MMSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_matmul_i8")), false);

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+i8mm"});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_matmul_i8")), true);
}

TEST_F(AProfileCanonicalizerTest, DefaultDotProdSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultDotProd) + "a";
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dotprod")), true);
}

TEST_F(AProfileCanonicalizerTest, DefaultDotProdSupportDisable) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultDotProd) + "a";
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "-dotprod"});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dotprod")), false);
}

using AProfileOptionalDotProd = AProfileCanonicalizerTestWithParam;
TEST_P(AProfileOptionalDotProd, OptionalDotProdSupport) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dotprod")), false);

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+dotprod"});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dotprod")), true);
}

TEST_F(AProfileCanonicalizerTest, ArchVersionInvalidLetter) {
  std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(defaultDotProd) + "b";
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dotprod")), false);
}

using AProfileOptionalSVE = AProfileCanonicalizerTestWithParam;
TEST_P(AProfileOptionalSVE, OptionalSVESupport) {
  const std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  EXPECT_TRUE(IsArch(target));
  EXPECT_FALSE(Downcast<Bool>(target.at("feature.has_sve")));

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_sve")));
}

TEST_F(AProfileCanonicalizerTest, DefaultSVESupportSVESupport) {
  const std::string arch_attr = "+v9a";

  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  EXPECT_TRUE(IsArch(target));
#if TVM_LLVM_VERSION >= 190 || (TVM_LLVM_VERSION / 10) == 13
  EXPECT_FALSE(Downcast<Bool>(target.at("feature.has_sve")));
#else
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_sve")));
#endif

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_sve")));
}

using AProfileOptionalFP16 = AProfileCanonicalizerTestWithParam;
TEST_P(AProfileOptionalFP16, OptionalFP16Support) {
  const std::string arch_attr = "+v" + FloatToStringWithoutTrailingZeros(GetParam()) + "a";

  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  EXPECT_TRUE(IsArch(target));
  EXPECT_FALSE(Downcast<Bool>(target.at("feature.has_fp16_simd")));

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+fullfp16"});
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_fp16_simd")));

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_fp16_simd")));
}

TEST_F(AProfileCanonicalizerTest, DefaultFP16Support) {
  const std::string arch_attr = "+v9a";

  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  EXPECT_TRUE(IsArch(target));
#if TVM_LLVM_VERSION >= 190 || (TVM_LLVM_VERSION / 10) == 13
  EXPECT_FALSE(Downcast<Bool>(target.at("feature.has_fp16_simd")));
#else
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_fp16_simd")));
#endif

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+fullfp16"});
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_fp16_simd")));

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sve"});
  EXPECT_TRUE(IsArch(target));
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_fp16_simd")));
}

TEST_F(AProfileCanonicalizerTest, ImpliedFeature) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-linux-gnu", {"+sve2"});
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_sve")));
  EXPECT_TRUE(Downcast<Bool>(target.at("feature.has_asimd")));
}

TEST_F(AProfileCanonicalizerTest, UnexpectedTargetKind) {
  EXPECT_THROW(
      {
        try {
          Canonicalize({{"kind", ffi::String("c")}});
        } catch (const tvm::ffi::Error& e) {
          EXPECT_THAT(e.what(), HasSubstr("Expected target kind 'llvm', but got 'c'"));
          throw;
        }
      },
      tvm::ffi::Error);
}

TEST(AProfileCanonicalizerInvalid, LLVMUnsupportedArchitecture) {
  if (has_aarch64_and_arm_targets) {
    GTEST_SKIP() << "LLVM has been compiled for the correct targets.";
  }
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({{"kind", ffi::String("llvm")}});
  // When LLVM doesn't support the architecture, no feature.* keys are set
  for (const auto& kv : target) {
    std::string key = kv.first;
    ASSERT_FALSE(key.substr(0, 8) == "feature.") << "Unexpected feature key: " << key;
  }
}

using AProfileOptionalSME = AProfileCanonicalizerTestWithParam;
TEST_P(AProfileOptionalSME, OptionalSMESupport) {
  const std::string arch_attr = "+v9a";

  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  ASSERT_TRUE(IsArch(target));
  ASSERT_FALSE(Downcast<Bool>(target.at("feature.has_sme")));

  target = CanonicalizeTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+sme"});
  ASSERT_TRUE(IsArch(target));
  ASSERT_TRUE(Downcast<Bool>(target.at("feature.has_sme")));
}

INSTANTIATE_TEST_SUITE_P(AProfileCanonicalizer, AProfileOptionalI8MM,
                         ::testing::ValuesIn(optionalI8MM));
INSTANTIATE_TEST_SUITE_P(AProfileCanonicalizer, AProfileOptionalDotProd,
                         ::testing::ValuesIn(optionalDotProd));
INSTANTIATE_TEST_SUITE_P(AProfileCanonicalizer, AProfileOptionalSVE,
                         ::testing::Values(8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9));
INSTANTIATE_TEST_SUITE_P(AProfileCanonicalizer, AProfileOptionalFP16,
                         ::testing::Values(8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9));
INSTANTIATE_TEST_SUITE_P(AProfileCanonicalizer, AProfileOptionalSME,
                         ::testing::ValuesIn(optionalSME));

}  // namespace aprofile
}  // namespace llvm
}  // namespace canonicalizer
}  // namespace target
}  // namespace tvm
