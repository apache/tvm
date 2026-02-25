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

#include "../src/target/canonicalizer/llvm/arm_mprofile.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace tvm {
namespace target {
namespace canonicalizer {
namespace llvm {
namespace mprofile {

static const char* mveCPUs[] = {"cortex-m55"};
static const char* dspCPUs[] = {"cortex-m4", "cortex-m7", "cortex-m33", "cortex-m35p"};
static const char* noExtensions[] = {"cortex-m0", "cortex-m3"};

class MProfileCanonicalizerMVECPUs : public testing::TestWithParam<const char*> {};
class MProfileCanonicalizerDSPCPUs : public testing::TestWithParam<const char*> {};
class MProfileCanonicalizerNoExtensions : public testing::TestWithParam<const char*> {};

static ffi::Map<ffi::String, ffi::Any> CanonicalizeTargetWithAttrs(ffi::String mcpu,
                                                                   ffi::Array<ffi::String> mattr) {
  return Canonicalize({{"mcpu", mcpu}, {"mattr", mattr}});
}

TEST(MProfileCanonicalizer, CheckIsNotArch) {
  ffi::String mcpu = "cake";
  ffi::Map<ffi::String, ffi::Any> fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), false);
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckIsArch) {
  ffi::String mcpu = GetParam();
  ffi::Map<ffi::String, ffi::Any> fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), true);
}

TEST_P(MProfileCanonicalizerDSPCPUs, CheckIsArch) {
  ffi::String mcpu = GetParam();
  ffi::Map<ffi::String, ffi::Any> fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), true);
}

TEST_P(MProfileCanonicalizerNoExtensions, CheckIsArch) {
  ffi::String mcpu = GetParam();
  ffi::Map<ffi::String, ffi::Any> fake_target = {{"mcpu", mcpu}};
  ASSERT_EQ(IsArch(fake_target), true);
}

TEST(MProfileCanonicalizer, CanonicalizeTarget) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({});
  ffi::Array<ffi::String> keys = Downcast<ffi::Array<ffi::String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "arm_cpu");
  ASSERT_EQ(keys[1], "cpu");

  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

TEST(MProfileCanonicalizer, CanonicalizeTargetWithExistingKeys) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({
      {"keys", ffi::Array<ffi::String>{"cpu"}},
  });
  ffi::Array<ffi::String> keys = Downcast<ffi::Array<ffi::String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST(MProfileCanonicalizer, CanonicalizeTargetWithDuplicateKey) {
  ffi::Map<ffi::String, ffi::Any> target = Canonicalize({
      {"keys", ffi::Array<ffi::String>{"cpu", "arm_cpu"}},
  });
  ffi::Array<ffi::String> keys = Downcast<ffi::Array<ffi::String>>(target.at("keys"));
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "arm_cpu");
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckMVESet) {
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(GetParam(), {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), true);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), true);
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckMVEOverrideCPU) {
  std::string mcpu = GetParam();
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(mcpu + "+nomve", {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), true);
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckDSPOverrideCPU) {
  std::string mcpu = GetParam();
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(mcpu + "+nodsp", {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckCombinedOverrideCPU) {
  std::string mcpu = GetParam();
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(mcpu + "+nodsp+nomve", {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
  target = CanonicalizeTargetWithAttrs(mcpu + "+nomve+nodsp", {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckMVEOverrideMAttr) {
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(GetParam(), {"+nomve"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), true);
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckDSPOverrideMattr) {
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(GetParam(), {"+nodsp"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

TEST_P(MProfileCanonicalizerMVECPUs, CheckCombinedOverrideMattr) {
  ffi::Map<ffi::String, ffi::Any> target =
      CanonicalizeTargetWithAttrs(GetParam(), {"+nodsp", "+nomve"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);

  target = CanonicalizeTargetWithAttrs(GetParam(), {"+nomve+nodsp"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);

  target = CanonicalizeTargetWithAttrs(GetParam(), {"+nomve", "+nodsp"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);

  target = CanonicalizeTargetWithAttrs(GetParam(), {"+woofles", "+nomve", "+nodsp"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

TEST_P(MProfileCanonicalizerDSPCPUs, CheckDSPSet) {
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(GetParam(), {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), true);
}

TEST_P(MProfileCanonicalizerDSPCPUs, CheckDSPOverrideCPU) {
  std::string mcpu = GetParam();
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(mcpu + "+nodsp", {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);

  target = CanonicalizeTargetWithAttrs(mcpu + "+nodsp+woofles", {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

TEST_P(MProfileCanonicalizerDSPCPUs, CheckDSPOverrideMattr) {
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(GetParam(), {"+nodsp"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);

  target = CanonicalizeTargetWithAttrs(GetParam(), {"+nodsp", "+woofles"});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

TEST_P(MProfileCanonicalizerNoExtensions, CheckNoFlags) {
  ffi::Map<ffi::String, ffi::Any> target = CanonicalizeTargetWithAttrs(GetParam(), {""});
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_mve")), false);
  ASSERT_EQ(Downcast<Bool>(target.at("feature.has_dsp")), false);
}

INSTANTIATE_TEST_CASE_P(MProfileCanonicalizer, MProfileCanonicalizerMVECPUs,
                        ::testing::ValuesIn(mveCPUs));
INSTANTIATE_TEST_CASE_P(MProfileCanonicalizer, MProfileCanonicalizerDSPCPUs,
                        ::testing::ValuesIn(dspCPUs));
INSTANTIATE_TEST_CASE_P(MProfileCanonicalizer, MProfileCanonicalizerNoExtensions,
                        ::testing::ValuesIn(noExtensions));

}  // namespace mprofile
}  // namespace llvm
}  // namespace canonicalizer
}  // namespace target
}  // namespace tvm
