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

/*!
 * \file src/contrib/ethosu/cascader/plan_generator.h
 * \brief Algorithm to generate possible Plans in the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_PLAN_GENERATOR_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_PLAN_GENERATOR_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

class CascaderGraph;
class MemoryRegion;
class Part;
class Tensor;
class StripeConfig;
class Plan;
class CascaderOptions;

using HomeMap =
    std::unordered_map<Tensor, std::vector<MemoryRegion>, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Generate possible output StripeConfigs that could be applied to a Part's output.
 * \param part The Part to generate StripeConfigs for.
 * \param stripe_factors How many striping factors to try per axis.
 * \param enable_striping Whether striping is enabled
 * \param multi_dimensional Whether to stripe in more than one dimension.
 * \return The generated StripeConfigs for the Part's output.
 */
std::vector<StripeConfig> GenerateOutputStripeConfigs(const Part& part, int stripe_factors,
                                                      bool enable_striping, bool multi_dimensional);

/*!
 * \brief Generate single-Part Plans for a Part for a given list of output StripeConfigs.
 * \param part The Part to generate Plans for.
 * \param output_stripe_configs The output StripeConfigs to generate Plans with.
 * \param home_map The Tensor homing map defining valid memory homes for Tensors.
 * \param options The configuration options with which to run the generator.
 * \return The generated Plans covering the Part.
 * \note For each of the output StripeConfigs provided, this algorithm will produce a number
 * of Plans corresponding to different choices of Tensor homing/copying, buffer modes
 * and INTERIOR/BOUNDARY states. For each of these variants, the Part's performance will
 * be queried and the memory usage will be calculated.
 */
std::vector<Plan> GenerateSinglePlans(const Part& part,
                                      const std::vector<StripeConfig>& output_stripe_configs,
                                      const HomeMap& home_map, const CascaderOptions& options);

/*!
 * \brief Generate pareto optimal Plans for a Graph.
 * \param graph The Graph to generate Plans for.
 * \param home_map The Tensor homing map defining valid memory homes for Tensors.
 * \param options The configuration options with which to run the generator.
 * \return A map between Part groups and a list of pareto optimal Plans which cover that group.
 * \note This algorithm does the following:
 *
 * Iterate Part-by-Part in a reversed topological ordering (starting at the output Parts and
 * working towards the input Parts).
 *
 * For each Part:
 *  1. Determine the possible StripeConfigs we might want to use to stripe the Part using
 *     GenerateOutputStripeConfigs.
 *  2. Additionally, collect all the StripeConfigs of open Plans that could connect to this
 *     Part (i.e. the Plan has an open TensorConfig for the Part's output Tensor).
 *  3. Use these two lists of StripeConfigs to produce single Part Plans with GenerateSinglePlans.
 *  4. For the generated Plans that have an open output TensorConfig, try and merge these into
 *     existing Plans which share an open input TensorConfig.
 *  5. All Plans are then indexed by both the Part group they cover and their open TensorConfigs.
 *  6. Plans which cover the same Part group and share the same open TensorConfigs are culled
 *     using ParetoCullPlans.
 *
 * Once every Part has been visited, return the Plans with no open TensorConfigs indexed by Part
 * group.
 */
std::unordered_map<std::vector<Part>, std::vector<Plan>> GenerateGraphPlans(
    const CascaderGraph& graph, const HomeMap& home_map, const CascaderOptions& options);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_PLAN_GENERATOR_H_
