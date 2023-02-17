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

#include "../../tir/schedule/analysis.h"
#include "../../tir/schedule/transform.h"
#include "../utils.h"
#include "multi_level_tiling_with_intrin.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::LoopRV;
using tir::Schedule;

class MultiLevelTilingHexagonNode : public MultiLevelTilingWithIntrinNode {
 private:
  // Subrule: Add software pipeline
  inline std::vector<State> AddSoftwarePipeline(State state) const;

  // Override ApplySubRules to apply tensorization-specific sub-rules
  std::vector<State> ApplySubRules(std::vector<State> states) final;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const override {
    ObjectPtr<MultiLevelTilingHexagonNode> n = make_object<MultiLevelTilingHexagonNode>(*this);
    return ScheduleRule(n);
  }

 public:
  /*! \brief Whether to use software pipeline */
  bool use_software_pipeline = false;
  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingHexagon";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingHexagonNode, MultiLevelTilingNode);
};

std::vector<State> MultiLevelTilingHexagonNode::ApplySubRules(std::vector<State> states) {
  states = MultiLevelTilingWithIntrinNode::ApplySubRules(states);
  states = SubRule(std::move(states), [&](State state) { return AddSoftwarePipeline(state); });
  return states;
}

std::vector<State> MultiLevelTilingHexagonNode::AddSoftwarePipeline(State state) const {
  if (!use_software_pipeline) {
    return {state};
  }
  // The current config is not suitable for software pipelining if the r_indices_ (reduction indicies) are less than 2.
  if (r_indices_.size() < 2) {
    return {state};
  }

  Schedule& sch = state->sch;
  // Check reduction length after blockize.
  int64_t reduction_length = 1;
  for (int r_index : r_indices_) {
    const Array<LoopRV>& tiles = state->tiles[r_index];
    for (const LoopRV& tile : tiles) {
      const auto* extent = sch->Get(tile)->extent.as<IntImmNode>();
      ICHECK(extent != nullptr) << "Dynamic extent is not supported.";
      reduction_length *= extent->value;
    }
  }
  if (reduction_length <= 1) {
    return {state};
  }

  // Return if there are more less than 1 or more than 2 cache_reads.
  size_t cache_read_count = state->read_reuse.size();
  if (cache_read_count > 2 || cache_read_count == 0) {
    return {state};
  }

  // Add annotations for software pipelining at the loop right above the cache read stages.
  Array<Integer> software_pipeline_stage;
  Array<Integer> software_pipeline_order;
  Array<Integer> software_pipeline_async_stages;
  if (cache_read_count == 2) {
    software_pipeline_stage = Array<Integer>{0, 0, 1}; // The pipeline merges the first 2 stages into one.
    software_pipeline_order = Array<Integer>{0, 1, 2};
    software_pipeline_async_stages = Array<Integer>{0}; // The 0th stage is set as async.
  } else {
    software_pipeline_stage = Array<Integer>{0, 1};
    software_pipeline_order = Array<Integer>{0, 1};
    software_pipeline_async_stages = Array<Integer>{0};
  }

  tir::BlockRV cache_read_block = state->read_reuse.begin()->second;
  Array<LoopRV> cache_read_loops = sch->GetLoops(cache_read_block);
  Array<LoopRV> reduction_loops;
  for (size_t i = 0; i < cache_read_loops.size() - 1; ++i) {
    if (tir::GetLoopIterType(sch->GetSRef(cache_read_loops[i])) != tir::IterVarType::kDataPar) {
      reduction_loops.push_back(cache_read_loops[i]);
    } else if (reduction_loops.size() > 0 &&
               sch->Get(cache_read_loops[i])->extent.as<IntImmNode>()->value == 1) {
      reduction_loops.push_back(cache_read_loops[i]);
    }
  }
  auto fused = sch->Fuse(reduction_loops);

  sch->Annotate(fused, tir::attr::software_pipeline_stage, software_pipeline_stage);
  sch->Annotate(fused, tir::attr::software_pipeline_order, software_pipeline_order);
  sch->Annotate(fused, tir::attr::software_pipeline_async_stages, software_pipeline_async_stages);

  // TODO(nverke): Add support for nested async pipelines.
  // TODO(nverke): Add support for async cache writes.

  return {state};
}

ScheduleRule ScheduleRule::MultiLevelTilingHexagon(
    Array<Map<String, String>> intrin_groups, String structure, Optional<Array<String>> tile_binds,
    Optional<Integer> max_innermost_factor, Optional<Array<Integer>> vector_load_lens,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write,
    bool use_software_pipeline) {
  CHECK(!tile_binds.defined()) << "Tile binds cannot be used on hexagon.";
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingHexagonNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);

  node->intrin_name = intrin_groups[0]["compute"];
  node->use_software_pipeline = use_software_pipeline;
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingHexagonNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingHexagon")
    .set_body_typed(ScheduleRule::MultiLevelTilingHexagon);

}  // namespace meta_schedule
}  // namespace tvm
