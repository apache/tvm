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

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace tvm {
namespace target {
namespace parsers {
namespace aprofile {

static float defaultI8MM = 8.6;
static float optionalI8MM[] = {8.2, 8.3, 8.4, 8.5};
static float defaultDotProd = 8.4;
static float optionalDotProd[] = {8.2, 8.3};

class AProfileOptionalI8MM : public testing::TestWithParam<float> {};
class AProfileOptionalDotProd : public testing::TestWithParam<float> {};

static TargetFeatures ParseTargetWithAttrs(String mcpu, String mtriple, Array<String> mattr) {
  return ParseTarget({
      {"mcpu", mcpu},
      {"mtriple", mtriple},
      {"mattr", mattr},
  });
}

TEST(AProfileParser, ParseTargetKeys) {
  TargetJSON target = ParseTarget({});
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 1);
  ASSERT_EQ(keys[0], "arm_cpu");
}

TEST(AProfileParser, ParseTargetWithExistingKeys) {
  TargetJSON target = ParseTarget({
      {"keys", Array<String>{"cpu"}},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST(AProfileParser, ParseTargetWithDuplicateKey) {
  TargetJSON target = ParseTarget({
      {"keys", Array<String>{"cpu", "arm_cpu"}},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST(AProfileParser, ParseTargetDefaults) {
  TargetJSON target = ParseTarget({});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));

  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), false);
}

TEST(AProfileParser, IsAArch64Triple) {
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("is_aarch64")), true);
}

TEST(AProfileParser, IsAArch32Triple) {
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

TEST(AProfileParser, IsAArch32BlankCPU) {
  TargetJSON target = ParseTarget({
      {"mtriple", String("arm-unknown-linux-gnu")},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
}

TEST(AProfileParser, IsAArch32TripleWithAProfile) {
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

TEST(AProfileParser, IsAArch32TripleWithMProfile) {
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

TEST(AProfileParser, AArch64HasASIMD) {
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);
}

TEST(AProfileParser, AArch32NoASIMD) {
  TargetJSON target = ParseTargetWithAttrs("", "armv8a-arm-none-eabi", {});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), false);
}

TEST(AProfileParser, AArch32HasASIMDWithOption) {
  TargetJSON target = ParseTargetWithAttrs("", "armv8a-arm-none-eabi", {"+simd"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);

  target = ParseTargetWithAttrs("cortex-a+simd", "armv8a-arm-none-eabi", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);
}

TEST(AProfileParser, AArch32HasASIMDWithAlternativeOption) {
  TargetJSON target = ParseTargetWithAttrs("", "armv8a-arm-none-eabi", {"+neon"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);

  target = ParseTargetWithAttrs("cortex-a+neon", "armv8a-arm-none-eabi", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_asimd")), true);
}

TEST(AProfileParser, NoI8MMSupport) {
  std::string attr = "+v8.0a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {attr, "+i8mm"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), false);
}

TEST(AProfileParser, DefaultI8MMSupport) {
  std::string arch_attr = "+v" + std::to_string(defaultI8MM) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), true);
}

TEST(AProfileParser, DefaultI8MMSupportDisable) {
  std::string arch_attr = "+v" + std::to_string(defaultI8MM) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+noi8mm"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), false);

  target = ParseTargetWithAttrs("cortex-a+noi8mm", "aarch64-arm-none-eabi", {arch_attr});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), false);
}

TEST_P(AProfileOptionalI8MM, OptionalI8MMSupport) {
  std::string arch_attr = "+v" + std::to_string(GetParam()) + "a";

  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), false);

  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+i8mm"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), true);

  target = ParseTargetWithAttrs("cortex-a+i8mm", "aarch64-arm-none-eabi", {arch_attr});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_matmul_i8")), true);
}

TEST(AProfileParser, NoDotProdSupport) {
  std::string attr = "+v8.0a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {attr, "+dotprod"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);
}

TEST(AProfileParser, DefaultDotProdSupport) {
  std::string arch_attr = "+v" + std::to_string(defaultDotProd) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), true);
}

TEST(AProfileParser, DefaultDotProdSupportDisable) {
  std::string arch_attr = "+v" + std::to_string(defaultDotProd) + "a";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+nodotprod"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);

  target = ParseTargetWithAttrs("cortex-a+nodotprod", "aarch64-arm-none-eabi", {arch_attr});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);
}

TEST_P(AProfileOptionalDotProd, OptionalDotProdSupport) {
  std::string arch_attr = "+v" + std::to_string(GetParam()) + "a";

  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);

  target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr, "+dotprod"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), true);

  target = ParseTargetWithAttrs("cortex-a+dotprod", "aarch64-arm-none-eabi", {arch_attr});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), true);
}

TEST(AProfileParser, ArchVersionInvalidLetter) {
  std::string arch_attr = "+v" + std::to_string(defaultDotProd) + "b";
  TargetJSON target = ParseTargetWithAttrs("", "aarch64-arm-none-eabi", {arch_attr});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(IsArch(target), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dotprod")), false);
}

INSTANTIATE_TEST_CASE_P(AProfileParser, AProfileOptionalI8MM, ::testing::ValuesIn(optionalI8MM));
INSTANTIATE_TEST_CASE_P(AProfileParser, AProfileOptionalDotProd,
                        ::testing::ValuesIn(optionalDotProd));

}  // namespace aprofile
}  // namespace parsers
}  // namespace target
}  // namespace tvm
