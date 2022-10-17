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
#include "plan.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "graph.h"
#include "tensor_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

void PlanNode::VisitAttrs(AttrVisitor* v) {
  Array<TensorConfig> tmp_arr(tensor_configs_);
  v->Visit("_tensor_configs", &tmp_arr);
  Array<TensorConfig> tmp_cfgs(open_configs_.begin(), open_configs_.end());
  v->Visit("_open_configs", &tmp_cfgs);
  v->Visit("_output_config", &output_config_);
  Array<Part> tmp_parts(part_group_.begin(), part_group_.end());
  v->Visit("_part_group", &tmp_parts);
  v->Visit("_interior_region", &interior_region_);
  v->Visit("_memory_usage", &memory_usage_);
  v->Visit("_cycles", &cycles_);
}

Plan::Plan(const std::vector<TensorConfig>& tensor_configs,
           const std::vector<TensorConfig>& open_configs, const TensorConfig& output_config,
           const std::vector<Part>& part_group, const MemoryRegion& interior_region,
           int memory_usage, int cycles) {
  auto n = make_object<PlanNode>();
  n->tensor_configs_ = std::move(tensor_configs);
  n->open_configs_ = std::move(open_configs);
  n->output_config_ = std::move(output_config);
  n->part_group_ = std::move(part_group);
  n->interior_region_ = interior_region;
  n->memory_usage_ = memory_usage;
  n->cycles_ = cycles;
  data_ = std::move(n);
}

Plan Plan::Merge(const Plan& other) const {
  auto n = make_object<PlanNode>(*this->operator->());
  n->tensor_configs_.insert(n->tensor_configs_.end(), other->tensor_configs_.begin(),
                            other->tensor_configs_.end());
  n->open_configs_.erase(
      std::remove(n->open_configs_.begin(), n->open_configs_.end(), (*this)->output_config_),
      n->open_configs_.end());
  for (const auto& config : other->open_configs_) {
    if (config->GetTensor() != (*this)->output_config_->GetTensor()) {
      n->open_configs_.push_back(config);
    }
  }
  n->output_config_ = other->output_config_;
  n->part_group_.insert(n->part_group_.end(), other->part_group_.begin(), other->part_group_.end());
  std::sort(n->part_group_.begin(), n->part_group_.end());
  n->memory_usage_ += other->memory_usage_;
  n->cycles_ += other->cycles_;
  return Plan(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.Plan")
    .set_body_typed([](Array<TensorConfig> tensor_configs, Array<TensorConfig> open_configs,
                       TensorConfig output_config, Array<Part> part_group,
                       MemoryRegion interior_region, int memory_usage, int cycles) {
      std::vector<TensorConfig> vtensor_configs(tensor_configs.begin(), tensor_configs.end());
      std::vector<TensorConfig> sopen_configs(open_configs.begin(), open_configs.end());
      std::vector<Part> spart_group(part_group.begin(), part_group.end());
      return Plan(vtensor_configs, sopen_configs, output_config, spart_group, interior_region,
                  memory_usage, cycles);
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PlanMerge").set_body_method(&Plan::Merge);

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.PlanMergeBenchmark")
    .set_body_typed([](Plan plan, Plan other, int repeats) {
      for (int i = 0; i < repeats; i++) {
        plan.Merge(other);
      }
      return plan.Merge(other);
    });

TVM_REGISTER_NODE_TYPE(PlanNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
