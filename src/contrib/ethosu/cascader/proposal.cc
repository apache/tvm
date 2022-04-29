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
#include "proposal.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "plan.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

void ProposalNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("_graph", &graph_);
  Array<Part> tmp_parts(part_group_.begin(), part_group_.end());
  v->Visit("_part_group", &tmp_parts);
  Array<Plan> tmp_plans(plans_.begin(), plans_.end());
  v->Visit("_plans", &tmp_plans);
  Map<Tensor, TensorConfig> tmp_tmap(input_tensor_configs_.begin(), input_tensor_configs_.end());
  v->Visit("_input_tensor_configs", &tmp_tmap);
  v->Visit("_cascade_region", &cascade_region_);
  v->Visit("_memory_usage", &memory_usage_);
  v->Visit("_cycles", &cycles_);
}

Proposal::Proposal(const CascaderGraph& graph, const std::vector<Part>& part_group,
                   const std::vector<Plan>& plans, const TensorConfigMap& input_tensor_configs,
                   const MemoryRegion& cascade_region, int memory_usage, int cycles) {
  auto n = make_object<ProposalNode>();
  n->graph_ = std::move(graph);
  n->part_group_ = std::move(part_group);
  std::sort(n->part_group_.begin(), n->part_group_.end());
  n->plans_ = std::move(plans);
  n->input_tensor_configs_ = std::move(input_tensor_configs);
  n->cascade_region_ = std::move(cascade_region);
  n->memory_usage_ = std::move(memory_usage);
  n->cycles_ = cycles;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.Proposal")
    .set_body_typed([](CascaderGraph graph, Array<Part> part_group, Array<Plan> plans,
                       Map<Tensor, TensorConfig> input_tensor_configs, MemoryRegion cascade_region,
                       int memory_usage, int cycles) {
      std::vector<Part> spart_group(part_group.begin(), part_group.end());
      std::vector<Plan> vplans(plans.begin(), plans.end());
      TensorConfigMap minput_tensor_configs(input_tensor_configs.begin(),
                                            input_tensor_configs.end());
      return Proposal(graph, spart_group, vplans, minput_tensor_configs, cascade_region,
                      memory_usage, cycles);
    });

TVM_REGISTER_NODE_TYPE(ProposalNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
