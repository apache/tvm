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
#include "inline.h"

#include <tvm/runtime/registry.h>

#include <utility>
#include <vector>

#include "../block_config.h"
#include "../common.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

const PerformanceInfo InlinePartNode::GetPerformanceInfo(const StripeConfig& output_stripe_config,
                                                         BufferMode buffer_mode) {
  std::vector<int64_t> read_bytes(input_tensors_.size());
  BlockConfig block_config = BlockConfig(std::vector<int>(1, 1), std::vector<int>(1, 1), 0, 0);
  PerformanceInfo info(0, read_bytes, 0, block_config);
  return info;
}

InlinePart::InlinePart(const TESubgraph& subgraph, const std::vector<Propagator> propagators) {
  auto n = make_object<InlinePartNode>();
  ICHECK_GT(propagators.size(), 0) << "The Part must include at least one Propagator.";
  n->subgraph_ = subgraph;
  n->propagators_ = std::move(propagators);
  n->in_line_ = true;
  n->input_tensors_.resize(propagators.size());
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.InlinePart")
    .set_body_typed([](Array<te::Tensor> subgraph_inputs, te::Tensor subgraph_output,
                       Array<Propagator> propagators) {
      std::vector<te::Tensor> vsubgraph_inputs(subgraph_inputs.begin(), subgraph_inputs.end());
      std::vector<Propagator> vpropagators(propagators.begin(), propagators.end());
      TESubgraph subgraph;
      subgraph.input_tensors = vsubgraph_inputs;
      subgraph.output_tensor = subgraph_output;
      return InlinePart(subgraph, vpropagators);
    });

TVM_REGISTER_NODE_TYPE(InlinePartNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
