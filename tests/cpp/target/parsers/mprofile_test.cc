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

#include "../src/target/parsers/mprofile.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace tvm {
namespace target {
namespace parsers {
namespace mprofile {

static const char* mveCPUs[] = {"cortex-m55"};
static const char* dspCPUs[] = {"cortex-m4", "cortex-m7", "cortex-m33", "cortex-m35p"};
static const char* noExtensions[] = {"cortex-m0", "cortex-m3"};

class MProfileParserMVECPUs : public testing::TestWithParam<const char*> {};
class MProfileParserDSPCPUs : public testing::TestWithParam<const char*> {};
class MProfileParserNoExtensions : public testing::TestWithParam<const char*> {};

static TargetFeatures ParseTargetWithAttrs(String mcpu, Array<String> mattr) {
  return ParseTarget({{"mcpu", mcpu}, {"mattr", mattr}});
}

TEST(MProfileParser, CheckIsNotArch) {
  String mcpu = "cake";
  TargetJSON fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), false);
}

TEST_P(MProfileParserMVECPUs, CheckIsArch) {
  String mcpu = GetParam();
  TargetJSON fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), true);
}

TEST_P(MProfileParserDSPCPUs, CheckIsArch) {
  String mcpu = GetParam();
  TargetJSON fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), true);
}

TEST_P(MProfileParserNoExtensions, CheckIsArch) {
  String mcpu = GetParam();
  TargetJSON fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), true);
}

TEST(MProfileParser, ParseTarget) {
  TargetJSON target = ParseTarget({});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "arm_cpu");
  ASSERT_EQ(keys[1], "cpu");

  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

TEST(MProfileParser, ParseTargetWithExistingKeys) {
  TargetJSON target = ParseTarget({
      {"keys", Array<String>{"cpu"}},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST(MProfileParser, ParseTargetWithDuplicateKey) {
  TargetJSON target = ParseTarget({
      {"keys", Array<String>{"cpu", "arm_cpu"}},
  });
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  Array<String> keys = Downcast<Array<String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST_P(MProfileParserMVECPUs, CheckMVESet) {
  TargetJSON target = ParseTargetWithAttrs(GetParam(), {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), true);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), true);
}

TEST_P(MProfileParserMVECPUs, CheckMVEOverrideCPU) {
  std::string mcpu = GetParam();
  TargetJSON target = ParseTargetWithAttrs(mcpu + "+nomve", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), true);
}

TEST_P(MProfileParserMVECPUs, CheckDSPOverrideCPU) {
  std::string mcpu = GetParam();
  TargetJSON target = ParseTargetWithAttrs(mcpu + "+nodsp", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

TEST_P(MProfileParserMVECPUs, CheckCombinedOverrideCPU) {
  std::string mcpu = GetParam();
  TargetJSON target = ParseTargetWithAttrs(mcpu + "+nodsp+nomve", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
  target = ParseTargetWithAttrs(mcpu + "+nomve+nodsp", {""});
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

TEST_P(MProfileParserMVECPUs, CheckMVEOverrideMAttr) {
  TargetJSON target = ParseTargetWithAttrs(GetParam(), {"+nomve"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), true);
}

TEST_P(MProfileParserMVECPUs, CheckDSPOverrideMattr) {
  TargetJSON target = ParseTargetWithAttrs(GetParam(), {"+nodsp"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

TEST_P(MProfileParserMVECPUs, CheckCombinedOverrideMattr) {
  TargetJSON target = ParseTargetWithAttrs(GetParam(), {"+nodsp", "+nomve"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);

  target = ParseTargetWithAttrs(GetParam(), {"+nomve+nodsp"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);

  target = ParseTargetWithAttrs(GetParam(), {"+nomve", "+nodsp"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);

  target = ParseTargetWithAttrs(GetParam(), {"+woofles", "+nomve", "+nodsp"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

TEST_P(MProfileParserDSPCPUs, CheckDSPSet) {
  TargetJSON target = ParseTargetWithAttrs(GetParam(), {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), true);
}

TEST_P(MProfileParserDSPCPUs, CheckDSPOverrideCPU) {
  std::string mcpu = GetParam();
  TargetJSON target = ParseTargetWithAttrs(mcpu + "+nodsp", {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);

  target = ParseTargetWithAttrs(mcpu + "+nodsp+woofles", {""});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

TEST_P(MProfileParserDSPCPUs, CheckDSPOverrideMattr) {
  TargetJSON target = ParseTargetWithAttrs(GetParam(), {"+nodsp"});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);

  target = ParseTargetWithAttrs(GetParam(), {"+nodsp", "+woofles"});
  features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

TEST_P(MProfileParserNoExtensions, CheckNoFlags) {
  TargetJSON target = ParseTargetWithAttrs(GetParam(), {""});
  TargetFeatures features = Downcast<TargetFeatures>(target.at("features"));
  ASSERT_EQ(Downcast<Bool>(features.at("has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(features.at("has_dsp")), false);
}

INSTANTIATE_TEST_CASE_P(MProfileParser, MProfileParserMVECPUs, ::testing::ValuesIn(mveCPUs));
INSTANTIATE_TEST_CASE_P(MProfileParser, MProfileParserDSPCPUs, ::testing::ValuesIn(dspCPUs));
INSTANTIATE_TEST_CASE_P(MProfileParser, MProfileParserNoExtensions,
                        ::testing::ValuesIn(noExtensions));

}  // namespace mprofile
}  // namespace parsers
}  // namespace target
}  // namespace tvm
