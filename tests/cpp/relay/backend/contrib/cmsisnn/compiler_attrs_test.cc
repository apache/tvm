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

#ifdef TVM_USE_CMSISNN

#include "../../../../../../src/relay/backend/contrib/cmsisnn/compiler_attrs.h"

#include <gtest/gtest.h>
#include <tvm/ir/transform.h>

#include <cmath>
#include <string>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

static const char* mveCPUs[] = {"cortex-m55"};
static const char* dspCPUs[] = {"cortex-m4", "cortex-m7", "cortex-m33", "cortex-m35p"};
static const char* noExtensions[] = {"cortex-m0", "cortex-m3"};

class CMSISNNFlagsMVECPUs : public testing::TestWithParam<const char*> {};
class CMSISNNFlagsDSPCPUs : public testing::TestWithParam<const char*> {};
class CMSISNNFlagsNoExtensions : public testing::TestWithParam<const char*> {};

static CMSISNNFlags GetFlagsWithCompilerAttrs(String mcpu, String mattr) {
  auto context_node = make_object<tvm::transform::PassContextNode>();
  auto cmsisnn_config_node = make_object<CMSISNNCompilerConfigNode>();
  cmsisnn_config_node->InitBySeq("mcpu", mcpu, "mattr", mattr);

  context_node->config = {
      {"relay.ext.cmsisnn.options", CMSISNNCompilerConfig(cmsisnn_config_node)}};

  tvm::transform::PassContext context = tvm::transform::PassContext(context_node);
  return GetCompilerFlags(context);
}

TEST(CMSISNNFlags, CreateFromUndefined) {
  auto context_node = make_object<tvm::transform::PassContextNode>();
  tvm::transform::PassContext context = tvm::transform::PassContext(context_node);
  CMSISNNFlags flags = GetCompilerFlags(context);
  ASSERT_EQ(flags.mve, false);
  ASSERT_EQ(flags.dsp, false);
}

TEST_P(CMSISNNFlagsMVECPUs, CheckMVESet) {
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(GetParam(), "");
  ASSERT_EQ(flags.dsp, true);
  ASSERT_EQ(flags.mve, true);
}

TEST_P(CMSISNNFlagsMVECPUs, CheckMVEOverrideCPU) {
  std::string mcpu = GetParam();
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(mcpu + "+nomve", "");
  ASSERT_EQ(flags.dsp, true);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsMVECPUs, CheckDSPOverrideCPU) {
  std::string mcpu = GetParam();
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(mcpu + "+nodsp", "");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsMVECPUs, CheckCombinedOverrideCPU) {
  std::string mcpu = GetParam();
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(mcpu + "+nodsp+nomve", "");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
  flags = GetFlagsWithCompilerAttrs(mcpu + "+nomve+nodsp", "");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsMVECPUs, CheckMVEOverrideMAttr) {
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(GetParam(), "+nomve");
  ASSERT_EQ(flags.dsp, true);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsMVECPUs, CheckDSPOverrideMattr) {
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(GetParam(), "+nodsp");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsMVECPUs, CheckCombinedOverrideMattr) {
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(GetParam(), "+nodsp+nomve");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
  flags = GetFlagsWithCompilerAttrs(GetParam(), "+nomve+nodsp");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
  flags = GetFlagsWithCompilerAttrs(GetParam(), "+woofles+nomve+nodsp");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsDSPCPUs, CheckDSPSet) {
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(GetParam(), "");
  ASSERT_EQ(flags.dsp, true);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsDSPCPUs, CheckDSPOverrideCPU) {
  std::string mcpu = GetParam();
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(mcpu + "+nodsp", "");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
  flags = GetFlagsWithCompilerAttrs(mcpu + "+nodsp+woofles", "");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsDSPCPUs, CheckDSPOverrideMattr) {
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(GetParam(), "+nodsp");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
  flags = GetFlagsWithCompilerAttrs(GetParam(), "+nodsp+woofles");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
}

TEST_P(CMSISNNFlagsNoExtensions, CheckNoFlags) {
  CMSISNNFlags flags = GetFlagsWithCompilerAttrs(GetParam(), "");
  ASSERT_EQ(flags.dsp, false);
  ASSERT_EQ(flags.mve, false);
}

INSTANTIATE_TEST_CASE_P(CMSISNNFlags, CMSISNNFlagsMVECPUs, ::testing::ValuesIn(mveCPUs));
INSTANTIATE_TEST_CASE_P(CMSISNNFlags, CMSISNNFlagsDSPCPUs, ::testing::ValuesIn(dspCPUs));
INSTANTIATE_TEST_CASE_P(CMSISNNFlags, CMSISNNFlagsNoExtensions, ::testing::ValuesIn(noExtensions));

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif
