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
#include <tvm/meta_schedule/schedule_rule.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../utils.h"
#include "./multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::LoopRV;
using tir::Schedule;

struct TensorCoreIntrinGroup {
  String init_intrin;
  String load_a_intrin;
  String load_b_intrin;
  String compute_intrin;
  String store_intrin;
};

class TensorCoreStateNode : public StateNode {
 public:
  /*! \brief The Tensor Core reindex block A for Tensor Core computation */
  tir::BlockRV tensor_core_reindex_A;
  /*! \brief The Tensor Core reindex block B for Tensor Core computation */
  tir::BlockRV tensor_core_reindex_B;
  /*! \brief The Tensor Core reindex store block for Tensor Core computation */
  tir::BlockRV tensor_core_reindex_store;

  State Copy() const final;

  static constexpr const char* _type_key = "meta_schedule.TensorCoreState";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorCoreStateNode, StateNode);
};

class TensorCoreState : public State {
 public:
  explicit TensorCoreState(tir::Schedule sch, tir::BlockRV block_rv,
                           Array<Array<tir::LoopRV>> tiles = {});

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TensorCoreState, State, TensorCoreStateNode);
};

TVM_REGISTER_OBJECT_TYPE(TensorCoreStateNode);

TensorCoreState::TensorCoreState(Schedule sch, BlockRV block_rv, Array<Array<LoopRV>> tiles) {
  ObjectPtr<TensorCoreStateNode> node = make_object<TensorCoreStateNode>();
  node->sch = std::move(sch);
  node->block_rv = std::move(block_rv);
  node->tiles = std::move(tiles);
  data_ = std::move(node);
}

State TensorCoreStateNode::Copy() const {
  ObjectPtr<TensorCoreStateNode> node = make_object<TensorCoreStateNode>(*this);
  node->sch = sch->Copy();
  return State(node);
}

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single group of tensor core
 * intrinsics.
 */
class MultiLevelTilingTensorCoreNode : public MultiLevelTilingNode {
 private:
  // SubRule: Add tensorization-related transformations
  inline std::vector<State> TransformForTensorization(TensorCoreState state) const;
  // Subrule: Add tensorized load
  inline std::vector<State> AddReadReuseTensorCore(TensorCoreState state) const;
  // Subrule: Add tensorized store
  inline std::vector<State> AddWriteReuseTensorCore(TensorCoreState state) const;

  // Override ApplySubRules to apply tensorization-specific sub-rules
  std::vector<State> ApplySubRules(std::vector<State> states) final;

  // Override Apply to apply tensorization-specific analysis before applying sub-rules
  Array<Schedule> Apply(const Schedule& sch, const BlockRV& block_rv) final;

  /*!
   * \brief Transform and tensorize with the given tensor intrin
   * \param state The state of the meta schedule rule
   * \param intrin_name The name of the tensor intrin
   * \return The loop to be tensorized. NullOpt if the workload can't be tensorized.
   */
  Optional<LoopRV> TransformWithTensorIntrin(TensorCoreStateNode* state,
                                             const String& intrin_name) const;

  /*!
   * \brief Tile, blockize and annotate for tensorization with the given intrin
   * \param block_rv The block to be tensorized
   * \param intrin_name The name of the tensor intrin
   */
  void TileAndAnnotateTensorize(Schedule* sch, const BlockRV& block_rv,
                                const String& intrin_name) const;

 public:
  /*! \brief The tensor core intrin group to apply */
  TensorCoreIntrinGroup intrin_group;
  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingTensorCore";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingTensorCoreNode, MultiLevelTilingNode);

 private:
  /*!
   * \brief The mapping info for auto tensorization
   */
  tir::AutoTensorizeMappingInfo mapping_info_{nullptr};
};

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingTensorCoreNode::Apply(const Schedule& sch,
                                                      const BlockRV& block_rv) {
  if (!NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
    return {sch};
  }

  Optional<tir::AutoTensorizeMappingInfo> mapping_info =
      tir::GetAutoTensorizeMappingInfo(sch->state(), sch->GetSRef(block_rv),
                                       tir::TensorIntrin::Get(intrin_group.compute_intrin)->desc);
  if (!mapping_info.defined()) {
    return {sch};
  }
  mapping_info_ = mapping_info.value();

  // Create a copy of the schedule so that we can roll back transformations if tensorization
  // fail.
  Schedule original_sch = sch->Copy();
  sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);

  Array<Schedule> results;
  for (auto&& state : ApplySubRules({TensorCoreState(sch, block_rv)})) {
    results.push_back(std::move(state->sch));
  }
  if (results.empty()) {
    return {original_sch};
  }
  return results;
}

std::vector<State> MultiLevelTilingTensorCoreNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    return TransformForTensorization(Downcast<TensorCoreState>(state));
  });
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(state); });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(state); });
  states = SubRule(std::move(states), [&](State state) {
    return AddWriteReuseTensorCore(Downcast<TensorCoreState>(state));
  });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(state); });
  states = SubRule(std::move(states), [&](State state) {
    return AddReadReuseTensorCore(Downcast<TensorCoreState>(state));
  });
  return states;
}

void MultiLevelTilingTensorCoreNode::TileAndAnnotateTensorize(Schedule* sch,
                                                              const BlockRV& block_rv,
                                                              const String& intrin_name) const {
  Optional<LoopRV> loop = TileWithTensorIntrin(*sch, block_rv, intrin_name).value();
  ICHECK(loop.defined());
  BlockRV blockized_outer = (*sch)->Blockize(loop.value());
  (*sch)->Annotate(blockized_outer, tir::attr::meta_schedule_auto_tensorize, intrin_name);
}

std::vector<State> MultiLevelTilingTensorCoreNode::AddWriteReuseTensorCore(
    TensorCoreState state) const {
  // Add the cache write stage for Tensor Core
  int level = r_indices_.front() - 1;
  const LoopRV& loop = state->tiles[level].back();
  Schedule& sch = state->sch;
  auto cache_write = sch->CacheWrite(state->block_rv, 0, "wmma.accumulator");
  sch->ReverseComputeAt(cache_write, loop, true);

  if (state->write_reuse.count(0)) {
    AnnotateCooperativeFetching(&sch, state->write_reuse[0]);
  }
  sch->ReverseComputeInline(state->tensor_core_reindex_store);
  TileAndAnnotateTensorize(&sch, cache_write, intrin_group.store_intrin);
  return {state};
}

std::vector<State> MultiLevelTilingTensorCoreNode::AddReadReuseTensorCore(
    TensorCoreState state) const {
  const Array<LoopRV>& r_tiles = state->tiles[r_indices_[1]];
  Schedule& sch = state->sch;
  ICHECK(!r_tiles.empty()) << "ValueError: Cannot find the suitable reduction loop in the block";

  auto f_tensorize_load = [&](int read_index, String scope, String intrin_name) {
    auto cache_read = sch->CacheRead(state->block_rv, read_index, scope);
    state->sch->ComputeAt(cache_read, r_tiles.back(), true);
    TileAndAnnotateTensorize(&sch, cache_read, intrin_name);
  };

  f_tensorize_load(0, "wmma.matrix_a", intrin_group.load_a_intrin);
  f_tensorize_load(1, "wmma.matrix_b", intrin_group.load_b_intrin);
  sch->ComputeInline(state->tensor_core_reindex_A);
  sch->ComputeInline(state->tensor_core_reindex_B);

  for (int i = 0; i < 2; ++i) {
    const tir::BlockRV cache_read = state->read_reuse.at(i);
    const tir::BlockNode* cache_read_block = sch->GetSRef(cache_read)->StmtAs<tir::BlockNode>();
    tir::Buffer cache_read_buffer = tir::GetNthAccessBuffer(
        sch->state(), GetRef<tir::Block>(cache_read_block), 0, tir::BufferIndexType::kWrite);
    const DataType& dtype = cache_read_buffer->dtype;
    if (dtype.is_float16()) {
      sch->StorageAlign(cache_read, 0, -2, 32, 8);
    } else if (dtype.is_int() && dtype.bits() == 8) {
      sch->StorageAlign(cache_read, 0, -2, 32, 16);
    } else {
      LOG(WARNING) << "StorageAlign is not applied for data type " << dtype
                   << ", shared memory accesses might be inefficient.";
    }
  }
  return {state};
}

Optional<LoopRV> MultiLevelTilingTensorCoreNode::TransformWithTensorIntrin(
    TensorCoreStateNode* state, const String& intrin_name) const {
  BlockRV block_rv = state->block_rv;
  tir::StmtSRef block_sref = state->sch->GetSRef(state->block_rv);

  // Add reindex stages
  const tir::BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Hold the reference of the block before reindex
  const tir::Block block_before_reindex = GetRef<tir::Block>(block);
  if (block->reads.size() != 2 || block->writes.size() != 1) {
    // only matmul-like computation is allowed
    return NullOpt;
  }
  state->tensor_core_reindex_store =
      state->sch->ReIndex(state->block_rv, 0, tir::BufferIndexType::kWrite);
  state->tensor_core_reindex_A =
      state->sch->ReIndex(state->block_rv, 0, tir::BufferIndexType::kRead);
  state->tensor_core_reindex_B =
      state->sch->ReIndex(state->block_rv, 1, tir::BufferIndexType::kRead);

  // Transform the layout of reindex buffers accordingly.
  // The index map defines the mapping for the computation block. We need to extract the sub index
  // map to transform the load and store block.
  ICHECK_EQ(mapping_info_->mappings.size(), 1U);  // assume only one mapping is present
  const tir::IndexMap& index_map = mapping_info_->mappings[0];

  // Find the correspondence between block iters and the iters in the index map.
  std::unordered_map<tir::Var, tir::Var, ObjectPtrHash, ObjectPtrEqual> lhs_to_index_map_src;
  std::unordered_map<tir::Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> rhs_to_index_map_tgt;
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> unmapped_index_map_src;
  ICHECK_EQ(mapping_info_->lhs_iters.size(), index_map->initial_indices.size());
  for (int i = 0; i < static_cast<int>(mapping_info_->lhs_iters.size()); ++i) {
    lhs_to_index_map_src[mapping_info_->lhs_iters[i]->var] = index_map->initial_indices[i];
  }
  // The number of result iters in the index map is equal or more than the number of rhs (the
  // tensor intrin) iters. When there are extra iters, these iters represent unmapped iters from the
  // lhs. They will be skipped during pattern matching for tensorization.
  // An example of such case is batch matmul, the batch dimension is kept after layout
  // transformations and it will be kept as a outer loop after tensorization.
  int offset = static_cast<int>(index_map->final_indices.size()) -
               static_cast<int>(mapping_info_->rhs_iters.size());
  ICHECK_GE(offset, 0);
  for (int i = 0; i < offset; ++i) {
    const tir::VarNode* var_ptr = index_map->final_indices[i].as<tir::VarNode>();
    ICHECK(var_ptr != nullptr);
    unmapped_index_map_src.insert(GetRef<tir::Var>(var_ptr));
  }
  for (int i = offset; i < static_cast<int>(index_map->final_indices.size()); ++i) {
    rhs_to_index_map_tgt[mapping_info_->rhs_iters[i - offset]->var] = index_map->final_indices[i];
  }

  auto f_get_sub_index_map = [&](const tir::Buffer& lhs_buffer, const tir::Region& lhs_region) {
    std::vector<tir::Var> sub_index_map_src;
    std::vector<PrimExpr> sub_index_map_tgt;
    const tir::Buffer& rhs_buffer = mapping_info_->lhs_buffer_map[lhs_buffer];
    for (const Range& range : lhs_region) {
      ICHECK(tir::is_one(range->extent));
      const tir::VarNode* var_ptr = range->min.as<tir::VarNode>();
      ICHECK(var_ptr != nullptr);
      const tir::Var& lhs_representer = lhs_to_index_map_src[GetRef<tir::Var>(var_ptr)];
      sub_index_map_src.push_back(lhs_representer);
      if (unmapped_index_map_src.count(lhs_representer)) {
        sub_index_map_tgt.push_back(lhs_representer);
      }
    }
    for (size_t i = 0; i < mapping_info_->rhs_buffer_indices[rhs_buffer].size(); ++i) {
      const tir::VarNode* var = mapping_info_->rhs_buffer_indices[rhs_buffer][i].as<tir::VarNode>();
      ICHECK(var != nullptr);
      sub_index_map_tgt.push_back(rhs_to_index_map_tgt[GetRef<tir::Var>(var)]);
    }
    return tir::IndexMap(sub_index_map_src, sub_index_map_tgt);
  };

  std::unordered_set<tir::Buffer, ObjectPtrHash, ObjectPtrEqual> visited_buffers;

  auto f_transform_buffer_layout = [&](tir::BufferIndexType index_type, int buffer_index) {
    const tir::Buffer& lhs_buffer = tir::GetNthAccessBuffer(
        state->sch->state(), block_before_reindex, buffer_index, index_type);
    if (visited_buffers.count(lhs_buffer)) {
      return;
    }
    visited_buffers.insert(lhs_buffer);
    // Refresh block pointer (block sref is not invalidated)
    block = TVM_SREF_TO_BLOCK(block, block_sref);
    const tir::BufferRegion& reindexed_buffer_region = tir::GetNthAccessBufferRegion(
        state->sch->state(), GetRef<tir::Block>(block), buffer_index, index_type);
    auto sub_index_map = f_get_sub_index_map(lhs_buffer, reindexed_buffer_region->region);
    state->sch->TransformLayout(state->block_rv, buffer_index, index_type, sub_index_map);
  };

  for (int i = 0, n = block_before_reindex->reads.size(); i < n; ++i) {
    f_transform_buffer_layout(tir::BufferIndexType::kRead, i);
  }
  for (int i = 0, n = block_before_reindex->writes.size(); i < n; ++i) {
    f_transform_buffer_layout(tir::BufferIndexType::kWrite, i);
  }

  // Transform the layout of current block and reindex blocks
  state->sch->TransformBlockLayout(state->tensor_core_reindex_store, index_map);
  state->sch->TransformBlockLayout(state->tensor_core_reindex_A, index_map);
  state->sch->TransformBlockLayout(state->tensor_core_reindex_B, index_map);
  state->sch->TransformBlockLayout(state->block_rv, index_map);

  return tir::TileWithTensorIntrin(state->sch, state->block_rv, intrin_name);
}

inline std::vector<State> MultiLevelTilingTensorCoreNode::TransformForTensorization(
    TensorCoreState state) const {
  // Do reindex and layout transformations.
  Optional<LoopRV> transformed_loop_rv =
      TransformWithTensorIntrin(state.operator->(), intrin_group.compute_intrin);
  if (!transformed_loop_rv.defined()) {
    // The workload can't be tensorized.
    return {};
  }

  // Do blockize
  state->block_rv = state->sch->Blockize(transformed_loop_rv.value());

  // Add annotations for post processors.
  state->sch->Annotate(state->block_rv, tir::attr::meta_schedule_auto_tensorize,
                       intrin_group.compute_intrin);
  state->sch->Annotate(state->block_rv, tir::attr::meta_schedule_auto_tensorize_init,
                       intrin_group.init_intrin);
  state->sch->Annotate(state->block_rv, tir::attr::warp_execution, Bool(true));
  return {std::move(state)};
}

ScheduleRule ScheduleRule::MultiLevelTilingTensorCore(
    Map<String, String> intrin_group, String structure, Optional<Array<String>> tile_binds,
    Optional<Integer> max_innermost_factor, Optional<Array<Integer>> vector_load_lens,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingTensorCoreNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);

  auto f_initialize_intrin = [&intrin_group](String key_name, String* intrin_name) {
    CHECK(intrin_group.count(key_name)) << "ValueError: " << key_name << " is not set.";
    *intrin_name = intrin_group.at(key_name);
    // Check the existence of the intrin
    tir::TensorIntrin::Get(*intrin_name);
  };
  f_initialize_intrin("init", &node->intrin_group.init_intrin);
  f_initialize_intrin("load_a", &node->intrin_group.load_a_intrin);
  f_initialize_intrin("load_b", &node->intrin_group.load_b_intrin);
  f_initialize_intrin("compute", &node->intrin_group.compute_intrin);
  f_initialize_intrin("store", &node->intrin_group.store_intrin);

  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingTensorCoreNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingTensorCore")
    .set_body_typed(ScheduleRule::MultiLevelTilingTensorCore);

}  // namespace meta_schedule
}  // namespace tvm
