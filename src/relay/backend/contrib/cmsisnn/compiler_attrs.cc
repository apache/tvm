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
#include "compiler_attrs.h"

#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <string>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

TVM_REGISTER_NODE_TYPE(CMSISNNCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.cmsisnn.options", CMSISNNCompilerConfig);

Target CreateTarget(const tvm::transform::PassContext& ctx) {
  auto cfg = ctx->GetConfig<CMSISNNCompilerConfig>("relay.ext.cmsisnn.options");
  if (!cfg.defined()) {
    return Target("cmsis-nn");
  }

  String mcpu = cfg.value()->mcpu;
  Array<String> mattr = {cfg.value()->mattr};
  Bool debug_last_error = cfg.value()->debug_last_error;

  Target cmsis_nn_target(TargetJSON{
      {"kind", String("cmsis-nn")},
      {"mcpu", mcpu},
      {"mattr", mattr},
      {"debug_last_error", debug_last_error},
  });

  return cmsis_nn_target;
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
