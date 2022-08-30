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
#include "ethosu.h"

#include <tvm/runtime/registry.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "../common.h"
#include "../stripe_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

const std::vector<int64_t> EthosuPartNode::GetBytesRead(const std::vector<int>& block_shape,
                                                        const std::vector<int>& full_shape) {
  std::vector<int64_t> bytes_per_input(propagators_.size(), 0);

  std::vector<int> order;
  std::vector<int> stripes;
  std::vector<int> offset;
  std::vector<float> strides;
  for (size_t i = 0; i < block_shape.size(); i++) {
    order.push_back(1);
    stripes.push_back(round_up_divide(full_shape[i], block_shape[i]));
    offset.push_back(0);
    strides.push_back(static_cast<float>(block_shape[i]));
  }

  StripeConfig output_block_config(block_shape, full_shape, strides, order, stripes, offset);
  auto input_block_configs = CalculateInputStripeConfigs(output_block_config);

  int i = 0;
  for (const auto& input_block_config : input_block_configs) {
    std::map<std::vector<int>, int> input_blocks = CountStripes(input_block_config, false);
    for (const auto& block : input_blocks) {
      bytes_per_input[i] +=
          mul_reduce(block.first) * block.second * input_tensors_[i]->GetDataType().bytes();
    }
    i++;
  }

  if (weight_tensor_idx_ != -1) {
    bytes_per_input[weight_tensor_idx_] *= (stripes[height_idx_] * stripes[width_idx_]);
  }

  return bytes_per_input;
}

float EthosuPartNode::CalculateCost(const BlockConfig& block_config,
                                    const StripeConfig& output_stripe_config) {
  std::vector<int> output_block = block_config->GetOutputBlockShape();
  std::vector<int> output_stripe_shape = output_stripe_config->GetShape();
  auto input_stripe_configs = CalculateInputStripeConfigs(output_stripe_config);
  std::vector<int> input_stripe_shape = input_stripe_configs[0]->GetShape();

  std::vector<int64_t> bytes_per_input = GetBytesRead(output_block, output_stripe_shape);
  bytes_per_input[0] *= subkernels_;

  // Calculate bytes read per output element
  float cost =
      static_cast<float>(bytes_per_input[0] + bytes_per_input[1]) / mul_reduce(output_stripe_shape);

  // Single buffering hardware optimization
  if (mul_reduce(input_stripe_shape) <= 2 * mul_reduce(block_config->GetInputBlockShape())) {
    cost /= 2;
  }
  return cost;
}

const BlockConfig EthosuPartNode::GetBlockConfig(const StripeConfig& output_stripe_config) {
  BlockConfig best_block_config = valid_block_configs_[0];
  float best_cost = CalculateCost(best_block_config, output_stripe_config);
  std::vector<int> output_stripe_shape = output_stripe_config->GetShape();
  auto input_stripe_configs = CalculateInputStripeConfigs(output_stripe_config);
  std::vector<int> input_stripe_shape = input_stripe_configs[0]->GetShape();

  for (const auto& block_config : valid_block_configs_) {
    float relative_cost = CalculateCost(block_config, output_stripe_config);
    if (relative_cost < best_cost) {
      best_block_config = block_config;
      best_cost = relative_cost;
    }
  }
  return best_block_config;
}

const PerformanceInfo EthosuPartNode::GetPerformanceInfo(const StripeConfig& output_stripe_config,
                                                         BufferMode buffer_mode) {
  BlockConfig block_config = GetBlockConfig(output_stripe_config);
  std::vector<int> block_shape = block_config->GetOutputBlockShape();

  std::vector<int64_t> bytes_per_input =
      GetBytesRead(block_shape, output_stripe_config->GetShape());

  float num_blocks = 1.0f;
  for (size_t i = 0; i < block_shape.size(); i++) {
    if (buffer_mode == BufferMode::RECOMPUTE) {
      num_blocks *= std::max(static_cast<float>(output_stripe_config->GetShape()[i]) /
                                 block_shape[i] * output_stripe_config->GetStripes()[i],
                             1.0f);
    } else {
      num_blocks *=
          std::max(static_cast<float>(output_tensor_->GetShape()[i]) / block_shape[i], 1.0f);
    }
  }

  float num_stripes = mul_reduce(output_stripe_config->GetStripes());
  std::vector<int64_t> read_bytes;
  for (int64_t stripe_bytes : bytes_per_input) {
    read_bytes.push_back(num_stripes * stripe_bytes);
  }
  int64_t write_bytes =
      num_blocks * mul_reduce(block_shape) * output_tensor_->GetDataType().bytes();

  int block_output_cycles = block_config->GetOutputCycles();
  int block_compute_cycles = block_config->GetComputeCycles();

  int64_t total_cycles = 0;
  if (block_output_cycles > block_compute_cycles) {
    total_cycles = (block_output_cycles * num_blocks) + block_compute_cycles;
  } else {
    total_cycles = (block_compute_cycles * num_blocks) + block_output_cycles;
  }

  PerformanceInfo info(total_cycles, read_bytes, write_bytes, block_config);
  return info;
}

EthosuPart::EthosuPart(const TESubgraph& subgraph, const std::vector<Propagator> propagators,
                       const std::vector<int>& output_quantum, int subkernels,
                       const std::vector<BlockConfig>& valid_block_configs, int weight_tensor_idx) {
  auto n = make_object<EthosuPartNode>();
  ICHECK_GT(propagators.size(), 0) << "The Part must include at least one Propagator.";
  n->subgraph_ = subgraph;
  n->propagators_ = std::move(propagators);
  n->in_line_ = false;
  n->input_tensors_.resize(propagators.size());
  n->output_quantum_ = output_quantum;
  n->valid_block_configs_ = valid_block_configs;
  n->subkernels_ = subkernels;
  n->weight_tensor_idx_ = weight_tensor_idx;
  if (output_quantum.size() == 5) {
    // NHCWB16 Format
    n->height_idx_ = 1;
    n->width_idx_ = 3;
  } else {
    // NHWC Format
    n->height_idx_ = 1;
    n->width_idx_ = 2;
  }
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.EthosuPart")
    .set_body_typed([](Array<te::Tensor> subgraph_inputs, te::Tensor subgraph_output,
                       Array<Propagator> propagators, Array<Integer> output_quantum, int subkernels,
                       Array<BlockConfig> valid_block_configs, int weight_tensor_idx) {
      std::vector<te::Tensor> vsubgraph_inputs(subgraph_inputs.begin(), subgraph_inputs.end());
      std::vector<Propagator> vpropagators(propagators.begin(), propagators.end());
      std::vector<int> voutput_quantum;
      std::transform(output_quantum.begin(), output_quantum.end(),
                     std::back_inserter(voutput_quantum),
                     [](auto&& val) { return val.IntValue(); });
      TESubgraph subgraph;
      subgraph.input_tensors = vsubgraph_inputs;
      subgraph.output_tensor = subgraph_output;
      std::vector<BlockConfig> vvalid_block_configs(valid_block_configs.begin(),
                                                    valid_block_configs.end());
      return EthosuPart(subgraph, vpropagators, voutput_quantum, subkernels, vvalid_block_configs,
                        weight_tensor_idx);
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.EthosuPartGetBlockConfig")
    .set_body_typed([](EthosuPart part, StripeConfig stripe_config) {
      return part->GetBlockConfig(stripe_config);
    });

TVM_REGISTER_NODE_TYPE(EthosuPartNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
