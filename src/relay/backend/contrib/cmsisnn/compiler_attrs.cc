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

#include <string>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

static const char* mveCPUs[] = {"cortex-m55"};
static const char* dspCPUs[] = {"cortex-m55", "cortex-m4", "cortex-m7", "cortex-m33",
                                "cortex-m35p"};

TVM_REGISTER_NODE_TYPE(CMSISNNCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.cmsisnn.options", CMSISNNCompilerConfig);

template <typename Container>
static inline bool MatchesCpu(std::string mcpu, const Container& cpus) {
  auto matches_cpu = [mcpu](const char* cpu) { return mcpu.find(cpu) == 0; };
  return std::find_if(std::begin(cpus), std::end(cpus), matches_cpu) != std::end(cpus);
}

static inline bool HasFlag(std::string attr, std::string flag) {
  return attr.find(flag) != std::string::npos;
}

CMSISNNFlags GetCompilerFlags(const tvm::transform::PassContext& ctx) {
  auto cfg = ctx->GetConfig<CMSISNNCompilerConfig>("relay.ext.cmsisnn.options");
  if (!cfg.defined()) {
    return kNoExt;
  }

  std::string mcpu = cfg.value()->mcpu;
  std::string mattr = cfg.value()->mattr;

  bool nomve = HasFlag(mcpu, "+nomve") || HasFlag(mattr, "+nomve");
  bool nodsp = HasFlag(mcpu, "+nodsp") || HasFlag(mattr, "+nodsp");

  auto has_mve = MatchesCpu(mcpu, mveCPUs);
  if (has_mve && !nomve && !nodsp) {
    return kHasMVE;
  }

  auto has_dsp = MatchesCpu(mcpu, dspCPUs);
  if (has_dsp && !nodsp) {
    return kHasDSP;
  }

  return kNoExt;
}

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
