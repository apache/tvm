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
#include <tvm/target/target.h>

#include <cmath>
#include <string>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

static Target GetTargetWithCompilerAttrs(String mcpu, String mattr) {
  auto context_node = make_object<tvm::transform::PassContextNode>();
  auto cmsisnn_config_node = make_object<CMSISNNCompilerConfigNode>();
  cmsisnn_config_node->InitBySeq("mcpu", mcpu, "mattr", mattr);

  context_node->config = {
      {"relay.ext.cmsisnn.options", CMSISNNCompilerConfig(cmsisnn_config_node)}};

  tvm::transform::PassContext context = tvm::transform::PassContext(context_node);
  return CreateTarget(context);
}

TEST(CMSISNNTarget, CreateFromUndefined) {
  auto context_node = make_object<tvm::transform::PassContextNode>();
  tvm::transform::PassContext context = tvm::transform::PassContext(context_node);
  Target target = CreateTarget(context);
  ASSERT_EQ(target->GetFeature<Bool>("has_mve").value_or(Bool(false)), Bool(false));
  ASSERT_EQ(target->GetFeature<Bool>("has_dsp").value_or(Bool(false)), Bool(false));
}

TEST(CMSISNNTarget, CreateFromContextCortexM55) {
  Target target = GetTargetWithCompilerAttrs("cortex-m55", "");
  ASSERT_EQ(target->GetFeature<Bool>("has_mve").value_or(Bool(false)), Bool(true));
  ASSERT_EQ(target->GetFeature<Bool>("has_dsp").value_or(Bool(false)), Bool(true));
}

TEST(CMSISNNTarget, CreateFromContextWithAttrsCortexM55) {
  Target target = GetTargetWithCompilerAttrs("cortex-m55", "+nomve");
  ASSERT_EQ(target->GetFeature<Bool>("has_mve").value_or(Bool(false)), Bool(false));
  ASSERT_EQ(target->GetFeature<Bool>("has_dsp").value_or(Bool(false)), Bool(true));
}

TEST(CMSISNNTarget, CreateFromContextCortexM85) {
  Target target = GetTargetWithCompilerAttrs("cortex-m85", "");
  ASSERT_EQ(target->GetFeature<Bool>("has_mve").value_or(Bool(false)), Bool(true));
  ASSERT_EQ(target->GetFeature<Bool>("has_dsp").value_or(Bool(false)), Bool(true));
}

TEST(CMSISNNTarget, CreateFromContextWithAttrsCortexM85) {
  Target target = GetTargetWithCompilerAttrs("cortex-m85", "+nomve");
  ASSERT_EQ(target->GetFeature<Bool>("has_mve").value_or(Bool(false)), Bool(false));
  ASSERT_EQ(target->GetFeature<Bool>("has_dsp").value_or(Bool(false)), Bool(true));
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif
