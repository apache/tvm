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

#include <tvm/ir/error.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../op/make_op.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosu {

/*! \brief Attributes to store the compiler options for Arm(R) Ethos(TM)-U NPU. */
struct EthosUCompilerConfigNode : public tvm::AttrsNode<EthosUCompilerConfigNode> {
  String accelerator_config;

  TVM_DECLARE_ATTRS(EthosUCompilerConfigNode, "ext.attrs.EthosUCompilerConfigNode") {
    TVM_ATTR_FIELD(accelerator_config)
        .describe(
            "The class of Arm(R) Ethos(TM)-U NPU; possible values = {ethos-u55-32, ethos-u55-64, "
            "ethos-u55-128, ethos-u55-256}")
        .set_default("ethos-u55-256");
  }
};

class EthosUCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(EthosUCompilerConfig, Attrs, EthosUCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(EthosUCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.ethosu.options", EthosUCompilerConfig);

auto GetCompilerAttrs() {
  auto ctx = transform::PassContext::Current();
  auto cfg = ctx->GetConfig<EthosUCompilerConfig>("relay.ext.ethosu.options");
  if (!cfg.defined()) {
    cfg = AttrsWithDefaultValues<EthosUCompilerConfig>();
  }
  return cfg;
}
TVM_REGISTER_GLOBAL("relay.ext.ethosu.get_compiler_attrs").set_body_typed(GetCompilerAttrs);

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
