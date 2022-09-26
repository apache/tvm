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
#include "multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Tile a subset of loops in the block according to the given tensor intrinsic, and annotate
 * the tiled block for tensorization by postproc rewrite.
 */
Optional<tir::BlockRV> TileForIntrin(tir::Schedule sch, tir::BlockRV block,
                                     const std::string& intrin_name) {
  Optional<tir::LoopRV> tiled_loop_rv = TileWithTensorIntrin(sch, block, intrin_name);
  if (!tiled_loop_rv) {
    return NullOpt;
  }
  ICHECK(tiled_loop_rv.defined());
  tir::BlockRV outer_block = sch->Blockize(tiled_loop_rv.value());
  sch->Annotate(outer_block, tir::attr::meta_schedule_auto_tensorize, String(intrin_name));
  return outer_block;
}

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single intrinsic.
 */
class MultiLevelTilingWithIntrinNode : public MultiLevelTilingNode {
 protected:
  Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
    auto desc_func = tir::TensorIntrin::Get(intrin_name).value()->desc;
    if (!CheckAutoTensorizeApplicable(sch, block_rv, desc_func)) {
      TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
      return {sch};
    }

    auto res = MultiLevelTilingNode::Apply(sch->Copy(), block_rv);

    if (res.empty()) {
      TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
      return {sch};
    }
    TVM_PY_LOG(INFO, logger) << "Tensorizing with " << intrin_name;
    return res;
  }

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<MultiLevelTilingWithIntrinNode> n =
        make_object<MultiLevelTilingWithIntrinNode>(*this);
    return ScheduleRule(n);
  }

  // Override ApplySubRules to tile the inner loops according to the given tensor intrinsic, then
  // tile the outerloops.
  virtual std::vector<State> ApplySubRules(std::vector<State> states) {
    states = SubRule(std::move(states), [&](State state) {
      if (auto block_rv = TileForIntrin(state->sch, state->block_rv, intrin_name)) {
        state->block_rv = block_rv.value();
        return std::vector<State>(1, state);
      }
      return std::vector<State>();
    });
    return MultiLevelTilingNode::ApplySubRules(states);
  }

 public:
  /*! \brief The name of a tensor intrinsic. */
  String intrin_name;

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingWithIntrin";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingWithIntrinNode, MultiLevelTilingNode);
};

ScheduleRule ScheduleRule::MultiLevelTilingWithIntrin(
    String intrin_name, String structure, Optional<Array<String>> tile_binds,
    Optional<Integer> max_innermost_factor, Optional<Array<Integer>> vector_load_lens,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write) {
  ICHECK(tir::TensorIntrin::Get(intrin_name).defined())
      << "Provided tensor intrinsic " << intrin_name << " is not registered.";
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingWithIntrinNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  node->intrin_name = intrin_name;
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingWithIntrinNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingWithIntrin")
    .set_body_typed(ScheduleRule::MultiLevelTilingWithIntrin);

}  // namespace meta_schedule
}  // namespace tvm
