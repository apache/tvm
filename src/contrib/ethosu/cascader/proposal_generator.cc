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
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cascader_options.h"
#include "graph.h"
#include "pareto.h"
#include "plan.h"
#include "plan_generator.h"
#include "proposal.h"
#include "stripe_config.h"
#include "tensor_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

std::unordered_set<TensorConfig> GetPlanBoundaryConfigs(const Plan& plan) {
  std::unordered_set<TensorConfig> boundary_configs;
  for (const auto& config : plan->GetTensorConfigs()) {
    if (config->GetState() == TensorConfigState::BOUNDARY) {
      boundary_configs.insert(config);
    }
  }
  return boundary_configs;
}

bool IsPlanCompatible(const Proposal& proposal, const std::vector<Part>& plan_part_group,
                      const std::unordered_set<TensorConfig>& plan_boundary_configs) {
  // Check the Plan Part group is disjoint with the Proposal Part group
  for (const auto& plan_part : plan_part_group) {
    for (const auto& proposal_part : proposal->GetPartGroup()) {
      if (plan_part == proposal_part) {
        return false;
      }
    }
  }
  // If the Plan and Proposal disagree on the memory home of a Tensor, they
  // are incompatible and can't be used to create a new Proposal
  auto tensor_configs = proposal->GetInputTensorConfigs();
  for (const auto& plan_config : plan_boundary_configs) {
    if (tensor_configs.find(plan_config->GetTensor()) != tensor_configs.end()) {
      auto proposal_config = tensor_configs.at(plan_config->GetTensor());
      if (proposal_config->GetHomeRegion() != plan_config->GetHomeRegion()) {
        return false;
      }
    }
  }
  return true;
}

std::unordered_map<Part, std::vector<Plan>, ObjectPtrHash, ObjectPtrEqual> CreatePlansByPart(
    const std::unordered_map<std::vector<Part>, std::vector<Plan>>& plans_by_group,
    const CascaderGraph& graph) {
  std::unordered_map<Part, std::vector<Plan>, ObjectPtrHash, ObjectPtrEqual> plans_by_part;
  for (const auto& it : plans_by_group) {
    auto part_group = it.first;
    auto plans = it.second;
    int highest_index = 0;
    Part& index_part = part_group.front();
    // Determine the Part in the Part group with the highest ID - this will be used to index
    // the Plans
    for (const auto& part : part_group) {
      int pid = graph->GetPartID(part);
      if (pid >= highest_index) {
        index_part = part;
        highest_index = pid;
      }
    }
    plans_by_part[index_part].insert(plans_by_part[index_part].begin(), plans.begin(), plans.end());
  }
  return plans_by_part;
}

Proposal AddPlanToProposal(const Proposal& proposal, const Plan& plan,
                           const std::unordered_set<TensorConfig>& plan_boundary_configs) {
  std::vector<Plan> new_plans = proposal->GetPlans();
  new_plans.push_back(plan);
  TensorConfigMap new_configs = proposal->GetInputTensorConfigs();
  // Add input configs from the Plan if they're homed in the cascade region
  for (const auto& config : plan_boundary_configs) {
    if (config->GetHomeRegion() == proposal->GetCascadeRegion()) {
      new_configs[config->GetTensor()] = config;
    }
  }
  // Remove the Plan's output config from the new_configs if it's present because
  // it won't be an input to the Proposal any more
  if (new_configs.find(plan->GetOutputConfig()->GetTensor()) != new_configs.end()) {
    new_configs.erase(plan->GetOutputConfig()->GetTensor());
  }
  // The updated memory usage is the memory required to run the Plan plus the
  // non-local memory that's required in the Proposal at that point in time
  int new_memory_usage = plan->GetMemoryUsage();
  for (const auto& it : new_configs) {
    if (plan_boundary_configs.find(it.second) == plan_boundary_configs.end()) {
      new_memory_usage += it.first->GetSize();
    }
  }
  new_memory_usage = std::max(new_memory_usage, proposal->GetMemoryUsage());
  int new_cycles = proposal->GetCycles() + plan->GetCycles();
  std::vector<Part> new_part_group = proposal->GetPartGroup();
  new_part_group.insert(new_part_group.end(), plan->GetPartGroup().begin(),
                        plan->GetPartGroup().end());
  std::sort(new_part_group.begin(), new_part_group.end());
  return Proposal(proposal->GetGraph(), new_part_group, new_plans, new_configs,
                  proposal->GetCascadeRegion(), new_memory_usage, new_cycles);
}

std::vector<Proposal> GeneratePartialProposals(
    const CascaderGraph& graph, const HomeMap& home_map, const CascaderOptions options,
    const std::unordered_map<Part, std::vector<Plan>, ObjectPtrHash, ObjectPtrEqual>& plans_by_part,
    const std::vector<Part>& partial_proposal_group,
    std::unordered_map<std::vector<Part>, std::vector<Proposal>>* proposals_by_group) {
  if (proposals_by_group->find(partial_proposal_group) != proposals_by_group->end()) {
    return proposals_by_group->at(partial_proposal_group);
  }
  if (partial_proposal_group.size() == 0) {
    (*proposals_by_group)[partial_proposal_group] =
        std::vector<Proposal>{Proposal(graph, std::vector<Part>(), std::vector<Plan>(),
                                       TensorConfigMap(), options->cascade_region, 0, 0)};
  } else {
    Part part = partial_proposal_group.back();
    const auto& plans = plans_by_part.at(part);
    for (const auto& plan : plans) {
      if (plan->GetInteriorRegion() == options->cascade_region) {
        // Doing this isn't very efficient, but it improves the performance of the Plan
        // generator
        std::unordered_set<TensorConfig> plan_boundary_configs = GetPlanBoundaryConfigs(plan);
        // The residual_proposal_group is a Part group indicating the Parts which aren't
        // covered by the current Plan. It's the group for which we must find 'residual
        // Proposals', meaning Proposals which cover the rest of the CascaderGraph assuming we
        // pick the current Plan.
        std::vector<Part> residual_proposal_group;
        std::copy_if(partial_proposal_group.begin(), partial_proposal_group.end(),
                     std::back_inserter(residual_proposal_group), [&plan](Part value) {
                       return std::find(plan->GetPartGroup().begin(), plan->GetPartGroup().end(),
                                        value) == plan->GetPartGroup().end();
                     });
        // std::sort(residual_proposal_group.begin(), residual_proposal_group.end());
        const auto& residual_proposals = GeneratePartialProposals(
            graph, home_map, options, plans_by_part, residual_proposal_group, proposals_by_group);
        auto plan_output_tensor = plan->GetOutputConfig()->GetTensor();
        ICHECK_LE(plan_output_tensor->GetProducers().size(), 1)
            << "All tensors must have at most one producer.";
        for (const auto& residual_proposal : residual_proposals) {
          if (IsPlanCompatible(residual_proposal, plan->GetPartGroup(), plan_boundary_configs)) {
            (*proposals_by_group)[partial_proposal_group].push_back(
                AddPlanToProposal(residual_proposal, plan, plan_boundary_configs));
          }
        }
      }
    }
    (*proposals_by_group)[partial_proposal_group] =
        ParetoCullProposals(proposals_by_group->at(partial_proposal_group), options->max_proposals,
                            options->disable_pareto_proposals);
  }
  return proposals_by_group->at(partial_proposal_group);
}

std::vector<Proposal> GenerateProposals(const CascaderGraph& graph, const HomeMap& home_map,
                                        const CascaderOptions options) {
  // First generate all the Pareto optimal Plans for the CascaderGraph
  auto plans_by_group = GenerateGraphPlans(graph, home_map, options);
  // First create a map between every Part in the CascaderGraph and all the Plans for which that
  // Part is the lowest ID Part within the Plan's Part group
  std::unordered_map<Part, std::vector<Plan>, ObjectPtrHash, ObjectPtrEqual> plans_by_part =
      CreatePlansByPart(plans_by_group, graph);
  // The Part group that partial Proposals are current being generated for
  std::vector<Part> partial_proposal_group = graph->GetPartOrder();
  // A map of Proposals indexed by the Part group they cover
  std::unordered_map<std::vector<Part>, std::vector<Proposal>> proposals_by_group;
  return GeneratePartialProposals(graph, home_map, options, plans_by_part, partial_proposal_group,
                                  &proposals_by_group);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.GenerateProposals")
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
      return Array<Proposal>(GenerateProposals(graph, mhome_map, options));
    });

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
