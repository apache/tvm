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

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/error.h>
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
  Bool enable_cascader = Bool(false);
  Bool enable_striping = Bool(false);
  Bool disable_copying_constants = Bool(false);
  String dev_force_block_config;
  String dev_max_open_plans;
  String dev_max_closed_plans;
  String dev_select_proposal_idx;
  Bool dev_disable_pareto_plans = Bool(false);
  Bool dev_disable_pareto_proposals = Bool(false);
  Bool dev_disable_block_culling = Bool(false);
  Bool dev_cascader_logging = Bool(false);

  TVM_DECLARE_ATTRS(EthosUCompilerConfigNode, "ext.attrs.EthosUCompilerConfigNode") {
    TVM_ATTR_FIELD(accelerator_config)
        .describe(
            "The class of Arm(R) Ethos(TM)-U NPU; possible values = {ethos-u55-32, ethos-u55-64, "
            "ethos-u55-128, ethos-u55-256}")
        .set_default("ethos-u55-256");
    TVM_ATTR_FIELD(enable_cascader)
        .describe("Whether the cascader should be enabled")
        .set_default(Bool(false));
    TVM_ATTR_FIELD(enable_striping)
        .describe("Whether the cascader should be striping")
        .set_default(Bool(false));
    TVM_ATTR_FIELD(disable_copying_constants)
        .describe(
            "Whether copying constants is disabled for case without the cascader. When this option "
            "is "
            "enabled, it is assumed that the constants should be located in SRAM (user determines "
            "in "
            "the linker script for section \".rodata.tvm\" that the constants are located in SRAM)")
        .set_default(Bool(false));
    String dev_warning = "Option is intended for development and debugging purposes only. ";
    TVM_ATTR_FIELD(dev_force_block_config)
        .describe((dev_warning + String("Force the block config to a given value; format = "
                                        "\"[BLK_HEIGHT]x[BLK_WIDTH]x[BLK_DEPTH]\""))
                      .data())
        .set_default("");
    TVM_ATTR_FIELD(dev_max_open_plans)
        .describe(
            (dev_warning + String("Specify the number of open plans kept for each part group"))
                .data())
        .set_default("8");
    TVM_ATTR_FIELD(dev_max_closed_plans)
        .describe(
            (dev_warning + String("Specify the number of closed plans kept for each part group"))
                .data())
        .set_default("32");
    TVM_ATTR_FIELD(dev_select_proposal_idx)
        .describe((dev_warning + String("Select proposal by index")).data())
        .set_default("-1");
    TVM_ATTR_FIELD(dev_disable_pareto_plans)
        .describe((dev_warning + String("Disable pareto culling for plans")).data())
        .set_default(Bool(false));
    TVM_ATTR_FIELD(dev_disable_pareto_proposals)
        .describe((dev_warning + String("Disable pareto culling for proposals")).data())
        .set_default(Bool(false));
    TVM_ATTR_FIELD(dev_disable_block_culling)
        .describe((dev_warning + String("Disable culling for block configs")).data())
        .set_default(Bool(false));
    TVM_ATTR_FIELD(dev_cascader_logging)
        .describe(
            (dev_warning + String("Enable cascader logging, log is dumped to .json file")).data())
        .set_default(Bool(false));
  }
};

class EthosUCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(EthosUCompilerConfig, Attrs, EthosUCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(EthosUCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.ethos-u.options", EthosUCompilerConfig);

auto GetCompilerAttrs() {
  auto ctx = transform::PassContext::Current();
  auto cfg = ctx->GetConfig<EthosUCompilerConfig>("relay.ext.ethos-u.options");
  if (!cfg.defined()) {
    cfg = AttrsWithDefaultValues<EthosUCompilerConfig>();
  }
  return cfg;
}
TVM_REGISTER_GLOBAL("relay.ext.ethos-u.get_compiler_attrs").set_body_typed(GetCompilerAttrs);

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
