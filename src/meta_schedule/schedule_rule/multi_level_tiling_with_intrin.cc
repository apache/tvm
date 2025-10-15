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

#include <tvm/ffi/reflection/registry.h>

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
ffi::Optional<tir::BlockRV> TileForIntrin(tir::Schedule sch, tir::BlockRV block,
                                          const std::string& intrin_name) {
  ffi::Optional<tir::LoopRV> tiled_loop_rv = TileWithTensorIntrin(sch, block, intrin_name);
  if (!tiled_loop_rv) {
    return std::nullopt;
  }
  ICHECK(tiled_loop_rv.defined());
  tir::BlockRV outer_block = sch->Blockize(tiled_loop_rv.value());
  sch->Annotate(outer_block, tir::attr::meta_schedule_auto_tensorize, ffi::String(intrin_name));
  return outer_block;
}

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single intrinsic.
 */
class MultiLevelTilingWithIntrinNode : public MultiLevelTilingNode {
 protected:
  ffi::Array<tir::Schedule> Apply(const tir::Schedule& sch, const tir::BlockRV& block_rv) final {
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
        ffi::make_object<MultiLevelTilingWithIntrinNode>(*this);
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
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MultiLevelTilingWithIntrinNode>();
  }

  /*! \brief The name of a tensor intrinsic. */
  ffi::String intrin_name;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.MultiLevelTilingWithIntrin",
                                    MultiLevelTilingWithIntrinNode, MultiLevelTilingNode);
};

ScheduleRule ScheduleRule::MultiLevelTilingWithIntrin(
    ffi::String intrin_name, ffi::String structure,
    ffi::Optional<ffi::Array<ffi::String>> tile_binds, ffi::Optional<Integer> max_innermost_factor,
    ffi::Optional<ffi::Array<Integer>> vector_load_lens,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_read,
    ffi::Optional<ffi::Map<ffi::String, ffi::Any>> reuse_write) {
  ICHECK(tir::TensorIntrin::Get(intrin_name).defined())
      << "Provided tensor intrinsic " << intrin_name << " is not registered.";
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingWithIntrinNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  node->intrin_name = intrin_name;
  return ScheduleRule(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MultiLevelTilingWithIntrinNode::RegisterReflection();
  refl::GlobalDef().def("meta_schedule.ScheduleRuleMultiLevelTilingWithIntrin",
                        ScheduleRule::MultiLevelTilingWithIntrin);
}

}  // namespace meta_schedule
}  // namespace tvm
