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
#include "plan_generator.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "block_config.h"
#include "cascader_options.h"
#include "common.h"
#include "graph.h"
#include "pareto.h"
#include "plan.h"
#include "stripe_config.h"
#include "tensor_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

template <class T>
std::vector<std::vector<T>> EnumerateCombinations(std::vector<std::vector<T>> values) {
  if (values.size() == 0) {
    return values;
  }
  if (values.size() == 1) {
    std::vector<std::vector<T>> combs;
    for (const auto& value : values[0]) {
      combs.push_back(std::vector<T>(1, value));
    }
    return combs;
  }
  auto combs = EnumerateCombinations(std::vector<std::vector<T>>(values.begin(), values.end() - 1));
  std::vector<std::vector<T>> new_combs;
  for (const auto& value : values.back()) {
    for (const auto& comb : combs) {
      auto new_comb = std::vector<T>(comb);
      new_comb.push_back(value);
      new_combs.push_back(new_comb);
    }
  }
  return new_combs;
}

float GetTransferEfficiency(const Tensor& tensor, const std::vector<int>& block_shape,
                            const MemoryRegion& memory) {
  // The block_shape represents the shape of the data transfer required for each job. This is used
  // to calculate how much of the block_shape is contiguous in memory (source memory for a read or
  // destination memory for a write) and subsequently calculate how efficient each memory burst is.
  const auto& shape = tensor->GetShape();
  int burst_length = block_shape[block_shape.size() - 1];
  if (block_shape[block_shape.size() - 1] == shape[shape.size() - 1]) {
    burst_length *= block_shape[block_shape.size() - 2];
  }

  burst_length *= tensor->GetDataType().bytes();
  return static_cast<float>(memory->burst_length) / std::min(burst_length, memory->burst_length);
}

std::vector<bool> GetCascadableAxes(const Part& part) {
  std::vector<bool> cascadable_axes(part->GetOutputTensor()->GetShape().size());
  // Check all the propagators to see if an output axis is projected into any
  // of the inputs. If it is, then that axis is cascadable.
  for (const auto& propagator : part->GetPropagators()) {
    auto transform = propagator->GetTransform();
    for (size_t i = 0; i < transform.size(); i++) {
      for (size_t j = 0; j < transform[0].size() - 1; j++) {
        // An axis is projected if there's a non-zero element
        // in the transform matrix
        if (transform[i][j] != 0) {
          cascadable_axes[j] = true;
        }
      }
    }
  }
  return cascadable_axes;
}

std::vector<StripeConfig> GenerateOutputStripeConfigs(const Part& part, int stripe_factors,
                                                      bool enable_striping,
                                                      bool multi_dimensional) {
  // If stripe_factors is <= 0, then we won't produce any StripeConfigs
  if (stripe_factors <= 0) {
    return std::vector<StripeConfig>();
  }
  // Work out the factors to divide by as inverse powers of 2.
  // The last factor is always reserved to be '0' which will correspond to
  // choosing a stripe size of 1 in the dimension. We always include this
  // as it represents the most extreme striping choice that uses the least
  // memory, so it is our choice of last resort.
  // For example, if stripe_factors = 4 then the factors are 1, 1/2, 1/4, 0.
  std::vector<float> factors;
  for (size_t i = 0; i < static_cast<size_t>(stripe_factors) - 1; i++) {
    factors.push_back(1.0f / (std::pow(2.0f, i)));
  }
  factors.push_back(0);
  // Then use the factors to derive the possible ways to split each dimension
  // into stripes. As an example, if an had extent 128 then by applying
  // the factors derived above we get the following possible splits for that axis:
  // 128, 64, 32, 1
  std::vector<std::vector<int>> splits;
  std::vector<int> output_shape = part->GetOutputTensor()->GetShape();
  size_t output_dims = output_shape.size();
  // Only bother striping along the axes which are cascadable
  auto cascadable_axes = GetCascadableAxes(part);
  for (size_t i = 0; i < output_dims; i++) {
    auto axis = output_shape[i];
    auto axis_align = part->GetStripeAlignHint()[i];
    std::set<int> axis_splits;  // Note this is a set to remove duplicate splits
    if (!cascadable_axes[i] || (!enable_striping)) {
      axis_splits.insert(axis);
    } else {
      for (float factor : factors) {
        int split =
            std::max(static_cast<int>(std::ceil(axis * factor / axis_align)), 1) * axis_align;
        split = std::min(axis, split);
        axis_splits.insert(split);
      }
    }
    splits.push_back(std::vector<int>(axis_splits.begin(), axis_splits.end()));
  }

  std::vector<std::vector<int>> stripe_shapes;
  if (multi_dimensional) {
    // Now calculate all the possible combinations of splits for each dimension
    // to give us all the possible stripe shapes. For example, if we had two axes
    // both with possible splits in {128, 64, 32, 1}, the stripe shapes would be:
    // (128, 128), (128, 64), (128, 32) ... (1, 64), (1, 32), (1, 1)
    stripe_shapes = EnumerateCombinations<int>(splits);
  } else {
    // Only consider splitting a single axis
    int axis = 0;
    for (const auto& split : splits) {
      for (const auto& axis_split : split) {
        std::vector<int> stripe_shape = output_shape;
        if (stripe_shape[axis] != axis_split) {
          stripe_shape[axis] = axis_split;
          stripe_shapes.push_back(stripe_shape);
        }
      }
      axis++;
    }
    stripe_shapes.push_back(output_shape);
  }
  auto offset = std::vector<int>(output_dims);
  std::vector<StripeConfig> stripe_configs;
  // Calculate the possible axis orderings such that each axis has the opportunity
  // to be the 'outermost' axis (which is axis that is chosen for rolling).
  std::vector<std::vector<int>> orders;
  for (size_t i = 0; i < output_dims; i++) {
    std::vector<int> order(output_dims);
    for (size_t j = 0; j < output_dims; j++) {
      order[j] = 1 + (j + i) % output_dims;
    }
    orders.push_back(order);
  }
  // Finally, create the StripeConfigs from the possible stripe shapes and orders
  for (const auto& stripe_shape : stripe_shapes) {
    std::vector<int> stripes;
    std::vector<float> strides;
    for (size_t i = 0; i < output_dims; i++) {
      stripes.push_back(std::ceil(static_cast<float>(output_shape[i]) / stripe_shape[i]));
      strides.push_back(static_cast<float>(stripe_shape[i]));  // strides = stripe_shape
    }
    // If the stripe shape equals the output shape (i.e. there's no striping), then
    // the order doesn't matter, so just pick the first order and continue.
    if (stripe_shape == output_shape) {
      stripe_configs.push_back(
          StripeConfig(stripe_shape, output_shape, strides, orders[0], stripes, offset));
      continue;
    }
    for (const auto& order : orders) {
      // Some logic to avoid having an axis be the 'outermost' if the stripe is full
      // size in that axis. This would otherwise be a waste because we can't roll
      // over an axis that hasn't been split.
      bool skip = false;
      for (size_t i = 0; i < output_dims; i++) {
        if (order[i] == 1 && stripe_shape[i] == output_shape[i]) {
          skip = true;
          break;
        }
      }
      if (skip) continue;
      stripe_configs.push_back(
          StripeConfig(stripe_shape, output_shape, strides, order, stripes, offset));
    }
  }
  return stripe_configs;
}

std::vector<TensorConfig> GetPossibleInputConfigs(const StripeConfig& stripe_config,
                                                  const Tensor& tensor,
                                                  const std::vector<MemoryRegion>& home_regions,
                                                  const CascaderOptions& options) {
  std::vector<TensorConfig> configs;
  for (const auto& home_region : home_regions) {
    // Boundary configs
    if (home_region == options->cascade_region || tensor->GetSize() > options->always_copy_size) {
      configs.push_back(TensorConfig(tensor, home_region, TensorConfigState::BOUNDARY,
                                     BufferMode::RECOMPUTE, {stripe_config}, false, home_region));
    }
    if (home_region != options->cascade_region) {
      configs.push_back(TensorConfig(tensor, home_region, TensorConfigState::BOUNDARY,
                                     BufferMode::ROLLING, {stripe_config}, true,
                                     options->cascade_region));
    }
  }
  if (!tensor->IsConstant()) {
    // Interior configs
    configs.push_back(TensorConfig(tensor, options->cascade_region, TensorConfigState::INTERIOR,
                                   BufferMode::RECOMPUTE, {stripe_config}, false,
                                   options->cascade_region));
    configs.push_back(TensorConfig(tensor, options->cascade_region, TensorConfigState::INTERIOR,
                                   BufferMode::ROLLING, {stripe_config}, false,
                                   options->cascade_region));
  }
  return configs;
}

// Check whether a StripeConfig can be an output boundary config
bool CanBound(const StripeConfig& stripe_config) {
  // Determine whether the StripeConfig results in non-overlapping stripes
  // which is the case when the stripe shape equals the strides
  for (size_t i = 0; i < stripe_config->GetShape().size(); i++) {
    // Check that the stripe shape and strides are equal
    if (stripe_config->GetShape()[i] - stripe_config->GetStrides()[i] != 0) {
      return false;
    }
  }
  return true;
}

std::vector<TensorConfig> GetPossibleOutputConfigs(const StripeConfig& stripe_config,
                                                   const Tensor& tensor,
                                                   const std::vector<MemoryRegion>& home_regions,
                                                   const CascaderOptions& options) {
  std::vector<TensorConfig> configs;
  // Only StripeConfigs with non-overlapping stripes can be output boundary configs
  if (CanBound(stripe_config)) {
    for (const auto& home_region : home_regions) {
      // Boundary configs
      configs.push_back(TensorConfig(tensor, home_region, TensorConfigState::BOUNDARY,
                                     BufferMode::RECOMPUTE, {stripe_config}, false, home_region));
    }
  }
  // Interior configs
  configs.push_back(TensorConfig(tensor, options->cascade_region, TensorConfigState::INTERIOR,
                                 BufferMode::RECOMPUTE, {stripe_config}, false,
                                 options->cascade_region));
  configs.push_back(TensorConfig(tensor, options->cascade_region, TensorConfigState::INTERIOR,
                                 BufferMode::ROLLING, {stripe_config}, false,
                                 options->cascade_region));
  return configs;
}

int GetInteriorMemoryUsage(const std::vector<TensorConfig>& input_configs,
                           const TensorConfig& output_config, const MemoryRegion& interior_region) {
  int memory_usage = 0;
  if (output_config->GetHomeRegion() == interior_region &&
      output_config->GetState() == TensorConfigState::BOUNDARY) {
    memory_usage += output_config->GetTensor()->GetSize();
  }
  for (const auto& input_config : input_configs) {
    if (input_config->GetHomeRegion() == interior_region &&
        input_config->GetState() == TensorConfigState::BOUNDARY) {
      memory_usage += input_config->GetTensor()->GetSize();
    } else if (input_config->GetHomeRegion() == interior_region ||
               input_config->GetCopyRegion() == interior_region) {
      memory_usage += input_config->GetBufferSize();
    }
  }
  return memory_usage;
}

/**
 * \brief Returns a hint estimating the number of cycles required for
 * the copy specified by tensor_config.
 *
 * \param tensor_config  The tensor configuration to estimate.
 * \return mem2mem_cycles Total estimated cycles.
 * \return initial_mem2mem_cycles Estimated cycles for the first block.
 */
std::pair<int, int> GetCopyCyclesHint(const TensorConfig& tensor_config) {
  Tensor tensor = tensor_config->GetTensor();
  MemoryRegion home_region = tensor_config->GetHomeRegion();
  MemoryRegion copy_region = tensor_config->GetCopyRegion();
  int initial_mem2mem_cycles = 0;
  int mem2mem_cycles = 0;

  // This Tensor needs to be copied - Count stripes for this config
  for (const auto& stripe_config : tensor_config->GetStripeConfigs()) {
    std::map<std::vector<int>, int> input_blocks = CountStripes(stripe_config, true);
    bool first_block = true;
    for (const auto& block : input_blocks) {
      int bytes_transferred = mul_reduce(block.first) * tensor->GetDataType().bytes() *
                              tensor->GetCompressionRatio() * block.second;
      int read_cycles = bytes_transferred * home_region->read_bandwidth + home_region->read_latency;
      int write_cycles = bytes_transferred * copy_region->write_bandwidth;

      if (first_block) {
        first_block = false;
        initial_mem2mem_cycles += std::max(read_cycles, write_cycles);
      }
      mem2mem_cycles += std::max(read_cycles, write_cycles);
    }
  }

  return {mem2mem_cycles, initial_mem2mem_cycles};
}

std::vector<Plan> GenerateSinglePlans(
    const Part& part, const std::vector<StripeConfig>& output_stripe_configs,
    const std::unordered_map<Tensor, std::vector<MemoryRegion>, ObjectPtrHash, ObjectPtrEqual>&
        home_map,
    const CascaderOptions& options) {
  std::vector<Plan> plans;
  std::vector<Part> part_group{part};
  // Create a selection of Plans per output_stripe_config
  for (const auto& output_stripe_config : output_stripe_configs) {
    // Calculate the input_stripe_configs
    auto input_stripe_configs = part->CalculateInputStripeConfigs(output_stripe_config);
    // From the input_stripe_configs, now derive all the possible input TensorConfigs
    std::vector<std::vector<TensorConfig>> all_possible_input_configs;
    size_t i = 0;
    for (const auto& stripe_config : input_stripe_configs) {
      Tensor tensor = part->GetInputTensors()[i];
      all_possible_input_configs.push_back(
          GetPossibleInputConfigs(stripe_config, tensor, home_map.at(tensor), options));
      i++;
    }
    // Now work out all the possible combinations of input TensorConfigs
    auto input_config_combinations =
        EnumerateCombinations<TensorConfig>(all_possible_input_configs);
    Tensor output_tensor = part->GetOutputTensor();
    // Then determine the possible output TensorConfigs (no combinations here because there's only
    // one output)
    auto output_configs = GetPossibleOutputConfigs(output_stripe_config, output_tensor,
                                                   home_map.at(output_tensor), options);
    // Calculate the performance information for the output_stripe_config for both the recompute and
    // rolling cases
    PerformanceInfo rolling_perf =
        part->GetPerformanceInfo(output_stripe_config, BufferMode::ROLLING);
    PerformanceInfo recompute_perf =
        part->GetPerformanceInfo(output_stripe_config, BufferMode::RECOMPUTE);
    // For all the possible input TensorConfig combinations
    for (const auto& input_configs : input_config_combinations) {
      std::vector<TensorConfig> tensor_configs;
      std::vector<TensorConfig> open_input_configs;
      // Add the input TensorConfigs to the 'tensor_configs' and
      // record which input TensorConfigs are 'open' (i.e. 'INTERIOR')
      for (const auto& input_config : input_configs) {
        tensor_configs.push_back(input_config);
        if (input_config->GetState() == TensorConfigState::INTERIOR) {
          open_input_configs.push_back(input_config);
        }
      }
      for (const auto& output_config : output_configs) {
        // Add the output TensorConfig to the tensor_configs and to
        // the open configs (if it's 'INTERIOR')
        tensor_configs.push_back(output_config);
        std::vector<TensorConfig> open_configs = open_input_configs;
        if (output_config->GetState() == TensorConfigState::INTERIOR) {
          open_configs.push_back(output_config);
        }
        int bandwidth_cycles = 0;
        int compute_cycles = 0;
        int mem2mem_cycles = 0;
        int initial_mem2mem_cycles = 0;

        // Pick the correct performance info based on the BufferMode
        PerformanceInfo perf_info;
        if (output_config->GetBufferMode() == BufferMode::RECOMPUTE) {
          perf_info = recompute_perf;
        } else {
          perf_info = rolling_perf;
        }
        // Calculate the bandwidth cycles by multiplying the bytes read/written by the
        // bandwidth of the memories
        BlockConfig block_config = perf_info->block_config;
        for (size_t i = 0; i < input_configs.size(); i++) {
          Tensor tensor = input_configs[i]->GetTensor();
          MemoryRegion copy_region = input_configs[i]->GetCopyRegion();

          if (input_configs[i]->DoCopy()) {
            std::pair<int, int> ret = GetCopyCyclesHint(input_configs[i]);
            mem2mem_cycles += ret.first;
            initial_mem2mem_cycles += ret.second;
          }
          float read_efficiency =
              GetTransferEfficiency(tensor, block_config->GetInputBlockShape(), copy_region);
          bandwidth_cycles +=
              (perf_info->read_bytes[i] / copy_region->read_bandwidth) * read_efficiency;
        }
        MemoryRegion write_region = output_config->GetCopyRegion();
        float write_efficiency = GetTransferEfficiency(
            output_config->GetTensor(), block_config->GetOutputBlockShape(), write_region);

        bandwidth_cycles +=
            perf_info->write_bytes / write_region->write_bandwidth * write_efficiency;
        compute_cycles = perf_info->compute_cycles;
        // Take the max of compute and bandwidth cycles as we assume compute cycles
        // can hide memory latency
        int cycles = std::max(std::max(compute_cycles, bandwidth_cycles), mem2mem_cycles);
        if (cycles > mem2mem_cycles) {
          // NPU cycles are the bottleneck - add initial mem2mem transfer cycles
          cycles += initial_mem2mem_cycles;
        }

        int memory_usage =
            GetInteriorMemoryUsage(input_configs, output_config, options->cascade_region);
        plans.push_back(Plan(tensor_configs, open_configs, output_config, part_group,
                             options->cascade_region, memory_usage, cycles));
      }
    }
  }
  return plans;
}

std::unordered_map<std::vector<Part>, std::vector<Plan>> GenerateGraphPlans(
    const CascaderGraph& graph,
    const std::unordered_map<Tensor, std::vector<MemoryRegion>, ObjectPtrHash, ObjectPtrEqual>&
        home_map,
    const CascaderOptions& options) {
  ICHECK_GT(options->stripe_factors, 0)
      << "stripe_factors = " << options->stripe_factors << ", but must be > 0";
  ICHECK_GT(options->max_plan_size, 0)
      << "max_plan_size = " << options->max_plan_size << ", but must be > 0";
  // Define a map between the graph Tensors and possible StripeConfigs that the Tensor may be
  // executed with
  std::unordered_map<Tensor, std::set<StripeConfig>, ObjectPtrHash, ObjectPtrEqual>
      stripe_configs_by_tensor;
  // Define a map between a given open TensorConfig and all the Plans which provide it
  std::unordered_map<TensorConfig, std::vector<Plan>> plans_by_config;
  // Define a map between a group of connected Parts and all the closed plans covering them
  std::unordered_map<std::vector<Part>, std::vector<Plan>> closed_plans;
  // Define a nested map which indexes open plans by both Part group and the open TensorConfigs they
  // provide. Note that we index in this way because Part group + open TensorConfigs combined
  // defines a group of Plans which can be mutually Pareto culled. If we culled of Part group alone,
  // we'd lose potentially valuable open Plans which could have gone on to be grown into Pareto
  // optimal closed plans.
  std::unordered_map<std::vector<Part>,
                     std::unordered_map<std::vector<TensorConfig>, std::vector<Plan>>>
      open_plans;
  // Traverse the graph in a reverse topological order (should be enforced by GetPartOrder)
  for (const auto& part : graph->GetPartOrder()) {
    // First generate all the possible StripeConfigs for the Part assuming that it will become the
    // output of a Plan. The number generated is a function of stripe_factors and the number of
    // cascadable dimensions in the Part.
    std::vector<StripeConfig> stripe_configs =
        GenerateOutputStripeConfigs(part, options->stripe_factors, options->enable_striping,
                                    options->enable_multi_dimensional_striping);
    // Check to see if the output Tensor is part of any existing open Plans
    if (stripe_configs_by_tensor.find(part->GetOutputTensor()) != stripe_configs_by_tensor.end()) {
      // If there are other open Plans which have this Part's output Tensor as an input, then
      // additionally consider the StripeConfigs of those open TensorConfigs so that we have the
      // option to merge into those open Plans.
      const std::set<StripeConfig>& connecting_configs =
          stripe_configs_by_tensor.at(part->GetOutputTensor());
      std::copy(connecting_configs.begin(), connecting_configs.end(),
                std::back_inserter(stripe_configs));
    }
    // Generate all the single Part Plans for the previously determined StripeConfigs
    auto single_part_plans = GenerateSinglePlans(part, stripe_configs, home_map, options);
    std::vector<Plan> plans;
    for (const auto& partial_plan : single_part_plans) {
      // If the output TensorConfig of the Plan is 'INTERIOR', then it must be merged with
      // another open Plan
      if (partial_plan->GetOutputConfig()->GetState() == TensorConfigState::INTERIOR) {
        if (plans_by_config.find(partial_plan->GetOutputConfig()) != plans_by_config.end() &&
            partial_plan->GetOutputConfig()->GetTensor()->GetConsumers().size() == 1) {
          // Search for all the open Plans which require the same TensorConfig
          const auto& join_plans = plans_by_config.at(partial_plan->GetOutputConfig());
          for (const auto& join_plan : join_plans) {
            // Only merge to form a new Plan if the resulting Plan size won't exceed the
            // max_plan_size
            if (join_plan->GetPartGroup().size() < static_cast<size_t>(options->max_plan_size)) {
              if (partial_plan->GetMemoryUsage() + join_plan->GetMemoryUsage() <
                  options->cascade_region->size) {
                plans.push_back(partial_plan.Merge(join_plan));
              }
            }
          }
        }
      } else {
        // If the single Part Plan had a 'BOUNDARY' output TensorConfig, then it doesn't need
        // merging and can stand on its own.
        plans.push_back(partial_plan);
      }
    }
    // For all the newly created Plans, update the various maps
    std::unordered_set<std::vector<Part>> new_part_groups;
    for (const auto& plan : plans) {
      new_part_groups.insert(plan->GetPartGroup());
      if (plan->IsClosed()) {
        closed_plans[plan->GetPartGroup()].push_back(plan);
      } else {
        open_plans[plan->GetPartGroup()][plan->GetOpenConfigs()].push_back(plan);
      }
    }
    // Now Pareto cull both the open and closed Plans to remove non-optimal Plans
    // Additionally, once culled we update another two maps, the stripe_configs_by_tensor
    // and plans_by_config maps.
    for (const auto& part_group : new_part_groups) {
      if (closed_plans.find(part_group) != closed_plans.end()) {
        closed_plans[part_group] = ParetoCullPlans(
            closed_plans.at(part_group), options->max_closed_plans, options->disable_pareto_plans);
      }
      for (const auto& it : open_plans[part_group]) {
        auto pareto_plans =
            ParetoCullPlans(it.second, options->max_open_plans, options->disable_pareto_plans);
        for (const auto& plan : pareto_plans) {
          for (const auto& open_config : plan->GetOpenConfigs()) {
            if (open_config != plan->GetOutputConfig()) {
              for (const auto& stripe_config : open_config->GetStripeConfigs()) {
                // Only add a StripeConfig if it contains for than one stripe
                if (mul_reduce(stripe_config->GetStripes()) > 1) {
                  stripe_configs_by_tensor[open_config->GetTensor()].insert(stripe_config);
                }
              }
              plans_by_config[open_config].push_back(plan);
            }
          }
        }
      }
    }
  }
  return closed_plans;
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.GenerateOutputStripeConfigs")
    .set_body_typed([](Part part, int stripe_factors, bool enable_striping,
                       bool multi_dimensional) {
      if (stripe_factors < 0) {
        return Array<StripeConfig>();
      }
      return Array<StripeConfig>(
          GenerateOutputStripeConfigs(part, stripe_factors, enable_striping, multi_dimensional));
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.GenerateSinglePlans")
    .set_body_typed([](Part part, Array<StripeConfig> output_stripe_configs,
                       Map<Tensor, Array<MemoryRegion>> home_map, CascaderOptions options) {
      std::vector<StripeConfig> voutput_stripe_configs(output_stripe_configs.begin(),
                                                       output_stripe_configs.end());
      std::unordered_map<Tensor, std::vector<MemoryRegion>, ObjectPtrHash, ObjectPtrEqual>
          mhome_map;
      for (const auto& it : home_map) {
        std::vector<MemoryRegion> home_regions;
        for (const auto& i : it.second) {
          home_regions.push_back(i);
        }
        mhome_map[it.first] = home_regions;
      }
      return Array<Plan>(GenerateSinglePlans(part, voutput_stripe_configs, mhome_map, options));
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.GenerateGraphPlans")
    .set_body_typed([](CascaderGraph graph, Map<Tensor, Array<MemoryRegion>> home_map,
                       CascaderOptions options) {
      std::unordered_map<Tensor, std::vector<MemoryRegion>, ObjectPtrHash, ObjectPtrEqual>
          mhome_map;
      for (const auto& it : home_map) {
        std::vector<MemoryRegion> home_regions;
        for (const auto& i : it.second) {
          home_regions.push_back(i);
        }
        mhome_map[it.first] = home_regions;
      }
      auto closed_plans = GenerateGraphPlans(graph, mhome_map, options);
      Map<Array<Part>, Array<Plan>> tclosed_plans;
      for (auto& it : closed_plans) {
        Array<Part> part_arr(it.first.begin(), it.first.end());
        Array<Plan> plan_arr(it.second);
        tclosed_plans.Set(part_arr, plan_arr);
      }
      return tclosed_plans;
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.GetCopyCyclesHint")
    .set_body_typed([](TensorConfig tensor_config) {
      std::pair<int, int> ret = GetCopyCyclesHint(tensor_config);
      return Array<Integer>({ret.first, ret.second});
    });

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
