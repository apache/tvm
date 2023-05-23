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
#include <tvm/tir/op.h>

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

  /*! \brief Create TensorCoreIntrinGroup from config in a map. The map should contains the
   * following keys:
   *  - init
   *  - load_a
   *  - load_b
   *  - compute
   *  - store
   * The values of the keys should be the names of the corresponding intrinsics and should be
   * registered via TensorIntrin.Register beforehand.
   */
  static TensorCoreIntrinGroup FromConfig(const Map<String, String>& config);
};

TensorCoreIntrinGroup TensorCoreIntrinGroup::FromConfig(const Map<String, String>& config) {
  auto f_initialize_intrin = [&config](String key_name, String* intrin_name) {
    CHECK(config.count(key_name)) << "ValueError: " << key_name << " is not set.";
    *intrin_name = config.at(key_name);
    // Check the existence of the intrin
    tir::TensorIntrin::Get(*intrin_name);
  };
  TensorCoreIntrinGroup intrin_group;
  f_initialize_intrin("init", &intrin_group.init_intrin);
  f_initialize_intrin("load_a", &intrin_group.load_a_intrin);
  f_initialize_intrin("load_b", &intrin_group.load_b_intrin);
  f_initialize_intrin("compute", &intrin_group.compute_intrin);
  f_initialize_intrin("store", &intrin_group.store_intrin);
  return intrin_group;
}

class TensorCoreStateNode : public StateNode {
 public:
  /*! \brief The tensor core intrinsic group. */
  TensorCoreIntrinGroup intrin_group;
  /*! \brief The auto tensorization maping info. */
  tir::AutoTensorizeMappingInfo mapping_info{nullptr};
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
  explicit TensorCoreState(TensorCoreIntrinGroup intrin_group,
                           tir::AutoTensorizeMappingInfo mapping_info, Schedule sch,
                           BlockRV block_rv, Array<Array<tir::LoopRV>> tiles = {});

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TensorCoreState, State, TensorCoreStateNode);
};

TVM_REGISTER_OBJECT_TYPE(TensorCoreStateNode);

TensorCoreState::TensorCoreState(TensorCoreIntrinGroup intrin_group,
                                 tir::AutoTensorizeMappingInfo mapping_info, Schedule sch,
                                 BlockRV block_rv, Array<Array<LoopRV>> tiles) {
  ObjectPtr<TensorCoreStateNode> node = make_object<TensorCoreStateNode>();
  node->intrin_group = intrin_group;
  node->mapping_info = mapping_info;
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
  // Subrule: Transform the layout of the output. This is necessary for efficient cache write the
  // output in the shared memory.
  std::vector<State> TransformIntermediateOutputLayout(TensorCoreState state);
  // Subrule: Add tensorized load
  inline std::vector<State> AddReadReuseTensorCore(TensorCoreState state) const;
  // Subrule: Add tensorized store
  inline std::vector<State> AddWriteReuseTensorCore(TensorCoreState state) const;
  // Subrule: Add software pipeline
  inline std::vector<State> AddSoftwarePipeline(TensorCoreState state) const;

  // Override ApplySubRules to apply tensorization-specific sub-rules
  std::vector<State> ApplySubRules(std::vector<State> states) final;

  // Override Apply to apply tensorization-specific analysis before applying sub-rules
  Array<Schedule> Apply(const Schedule& sch, const BlockRV& block_rv) final;

  // Inherited from ScheduleRuleNode
  ScheduleRule Clone() const final {
    ObjectPtr<MultiLevelTilingTensorCoreNode> n =
        make_object<MultiLevelTilingTensorCoreNode>(*this);
    return ScheduleRule(n);
  }

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
  /*! \brief The candidate tensor core intrin groups to apply */
  std::vector<TensorCoreIntrinGroup> intrin_groups;
  /*! \brief Whether to use software pipeline */
  bool use_software_pipeline = false;
  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingTensorCore";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingTensorCoreNode, MultiLevelTilingNode);

 private:
};

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingTensorCoreNode::Apply(const Schedule& sch,
                                                      const BlockRV& block_rv) {
  if (!NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
    return {sch};
  }

  std::unordered_map<int, tir::AutoTensorizeMappingInfo> intrin_group_to_mapping_info;
  for (int i = 0, n = intrin_groups.size(); i < n; ++i) {
    TensorCoreIntrinGroup intrin_group = intrin_groups[i];
    Optional<tir::AutoTensorizeMappingInfo> mapping_info = tir::GetAutoTensorizeMappingInfo(
        sch->state(), sch->GetSRef(block_rv),
        tir::TensorIntrin::Get(intrin_groups[i].compute_intrin).value()->desc);
    if (mapping_info.defined()) {
      intrin_group_to_mapping_info.emplace(i, mapping_info.value());
    }
  }

  if (intrin_group_to_mapping_info.empty()) {
    // No tensor intrinsics can be applied.
    return {sch};
  }

  // Save the original schedule so that we can roll back transformations if tensorization
  // fail.
  Schedule original_sch = sch;

  std::vector<State> initial_states;
  for (const auto& kv : intrin_group_to_mapping_info) {
    const TensorCoreIntrinGroup& intrin_group = intrin_groups[kv.first];
    const tir::AutoTensorizeMappingInfo& mapping_info = kv.second;
    Schedule new_sch = sch->Copy();
    new_sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);
    initial_states.push_back(TensorCoreState(intrin_group, mapping_info, new_sch, block_rv));
  }
  Array<Schedule> results;
  for (auto&& state : ApplySubRules(initial_states)) {
    TVM_PY_LOG(INFO, logger) << "Sketch " << results.size() << ": tensorizing with "
                             << state.as<TensorCoreStateNode>()->intrin_group.compute_intrin;
    results.push_back(std::move(state->sch));
  }
  if (results.empty()) {
    TVM_PY_LOG(INFO, logger) << "The workload cannot be tensorized.";
    return {original_sch};
  }
  return results;
}

std::vector<State> MultiLevelTilingTensorCoreNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) {
    return TransformForTensorization(Downcast<TensorCoreState>(state));
  });
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(state); });
  states = SubRule(std::move(states), [&](State state) {
    return TransformIntermediateOutputLayout(Downcast<TensorCoreState>(state));
  });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(state); });
  states = SubRule(std::move(states), [&](State state) {
    return AddWriteReuseTensorCore(Downcast<TensorCoreState>(state));
  });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(state); });
  states = SubRule(std::move(states), [&](State state) {
    return AddReadReuseTensorCore(Downcast<TensorCoreState>(state));
  });
  states = SubRule(std::move(states), [&](State state) {
    return AddSoftwarePipeline(Downcast<TensorCoreState>(state));
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

std::vector<State> MultiLevelTilingTensorCoreNode::TransformIntermediateOutputLayout(
    TensorCoreState state) {
  // Transform the intermediate output to packed layout
  //   [..., warp_m, warp_n, accum_frag_m, accum_frag_n, accum_elem_m, accum_elem_n]
  // where warp_m, warp_n are thread indices bound to the warp id, accum_frag_m, accum_frag_n are
  // the index of the fragments in each warp, accum_elem_m, accum_elem_n are the index of the
  // elements in each accumulator fragment.

  // Get the shape of the wmma accumulator
  auto [frag_shape_m, frag_shape_n] = [&]() {
    tir::Block intrin_block =
        Downcast<tir::BlockRealize>(
            tir::TensorIntrin::Get(state->intrin_group.init_intrin).value()->desc->body)
            ->block;
    tir::For loop_m = Downcast<tir::For>(intrin_block->body);
    tir::For loop_n = Downcast<tir::For>(loop_m->body);
    return std::make_tuple(loop_m->extent, loop_n->extent);
  }();

  // Get the tile index of the warp id (i.e. threadIdx.y)
  auto it = std::find(tile_binds.begin(), tile_binds.end(), "threadIdx.y");
  ICHECK(it != tile_binds.end());
  auto tile_index_warp_id = std::distance(tile_binds.begin(), it);

  // Get the extent of loop indicated by `loop_idx` inside the warp scope.
  // For example, after spatial loops i, j are tiled, we will have
  // tile_factors = ((i0, j0), (i1, j1), ..., (in, jn))
  // This function computes the product of tile_factors[i][loop_idx] for i > tile_index_warp_id.
  // `loop_idx` can be negative, in which case it is counted from the end.
  auto f_get_inner_tile_product = [&](int loop_idx) {
    Array<tir::ExprRV> factors;
    for (int i = tile_index_warp_id + 1; i < static_cast<int>(s_indices_.size()); ++i) {
      auto s_factors = state->tile_factors[s_indices_[i]];
      if (loop_idx < 0) {
        loop_idx += s_factors.size();
      }
      factors.push_back(s_factors[loop_idx]);
    }
    ICHECK(!factors.empty());
    if (factors.size() == 1) {
      return factors[0];
    }
    auto result = factors[0];
    for (int i = 1; i < static_cast<int>(factors.size()); ++i) {
      result = result * factors[i];
    }
    return result;
  };

  // Compute the number of output fragment of each warp
  auto warp_num_frag_m = f_get_inner_tile_product(-2);
  auto warp_num_frag_n = f_get_inner_tile_product(-1);

  Schedule& sch = state->sch;
  int buffer_ndim = static_cast<int>(sch->Get(state->block_rv)->writes[0]->buffer->shape.size());
  // The dimension of the buffer should be larger or same as that of the tensor intrin.
  ICHECK_GE(buffer_ndim, 2);
  int num_higher_dims = buffer_ndim - 2;

  auto index_map =
      tir::IndexMap::FromFunc(buffer_ndim,
                              // frag_shape_m and frag_shape_n are structural bindings that cannot
                              // not be automatically captured until c++20
                              [&, frag_shape_m = frag_shape_m,
                               frag_shape_n = frag_shape_n](const Array<tir::Var>& indices) {
                                Array<PrimExpr> result;
                                result.reserve(indices.size() + 4);
                                for (int i = 0; i < num_higher_dims; ++i) {
                                  result.push_back(indices[i]);
                                }
                                const auto& m = indices[num_higher_dims];
                                const auto& n = indices[num_higher_dims + 1];
                                auto accum_m = floormod(m, frag_shape_m);
                                auto accum_n = floormod(n, frag_shape_n);
                                auto outer_m = floordiv(m, frag_shape_m);
                                auto outer_n = floordiv(n, frag_shape_n);

                                result.push_back(floordiv(outer_m, warp_num_frag_m));
                                result.push_back(floordiv(outer_n, warp_num_frag_n));
                                result.push_back(floormod(outer_m, warp_num_frag_m));
                                result.push_back(floormod(outer_n, warp_num_frag_n));
                                result.push_back(accum_m);
                                result.push_back(accum_n);
                                return result;
                              });
  sch->TransformLayout(state->block_rv, 0, tir::BufferIndexType::kWrite, index_map,
                       /*pad_value=*/NullOpt, /*assume_injective_transform=*/true);

  return {state};
}

std::vector<State> MultiLevelTilingTensorCoreNode::AddWriteReuseTensorCore(
    TensorCoreState state) const {
  // Add the cache write stage for Tensor Core
  Schedule& sch = state->sch;
  auto cache_write = sch->CacheWrite(state->block_rv, 0, "wmma.accumulator");

  // The compute block has been tiled by the warp shape and the fragment shape.
  // We need to bind the cache write block (from the accumulator to the shared memory) to the warp
  // id. The schedule is as follows:
  //
  // After adding cache write for wmma.accumulator, we will have
  //   for i0, j0, i1, j1, accum_m, accum_n:
  //     shared_mem[i0, j0, i1, j1, accum_m, accum_n] = accum[i0, j0, i1, j1, accum_m, accum_n]
  //   for i0', j0', i1', j1', accum_m', accum_n':
  //      global_mem[i0', j0', i1', j1', accum_m', accum_n'] =
  //        shared_mem[i0', j0', i1', j1', accum_m', accum_n']
  // where i0' and j0' are already bound to the block id and warp id.
  //
  // To reduce the shared memory usage and allow efficient data movement, we will apply
  // transformations to generate the following schedule:
  //
  //   for i1':
  //     for i0_j0 (fused and bound to threadIdx.y):
  //       for j1, accum_m, accum_n:
  //         shared_mem[i0, j0, i1, j1, accum_m, accum_n] = accum[i0, j0, i1, j1, accum_m, accum_n]
  //     for i0', j0', j1', accum_m', accum_n':
  //       global_mem[i0', j0', i1', j1', accum_m', accum_n'] =
  //         shared_mem[i0', j0', i1', j1', accum_m', accum_n']
  //
  // i1' is reordered to the outermost. This effectively allows only a row (i.e. loop i1') of the
  // fragments are moved to the shared memory and then to the global memory each time.
  // As a result, shared memory for the output will only have shape of [j1, accum_m, accum_n]
  // instead of [i0 * i1 * accum_m, j0 * j1 * accum_n].

  // Get the loops other than the innermost two loops (accum_m and accum_n).
  auto f_get_loops = [&](const BlockRV& block_rv) -> std::array<LoopRV, 4> {
    Array<LoopRV> buffer_loops = sch->GetLoops(block_rv);
    ICHECK_GT(buffer_loops.size(), 6);
    return {buffer_loops[buffer_loops.size() - 6], buffer_loops[buffer_loops.size() - 5],
            buffer_loops[buffer_loops.size() - 4], buffer_loops[buffer_loops.size() - 3]};
  };
  {
    const auto& [i0, j0, i1, j1] = f_get_loops(state->write_reuse[0]);
    sch->Reorder({i1, i0, j0, j1});
    sch->ComputeAt(cache_write, i1, true);
  }
  {
    auto loops = f_get_loops(cache_write);
    const auto& i0 = loops[0];
    const auto& j0 = loops[1];
    auto fused = sch->Fuse({i0, j0});
    sch->Bind(fused, "threadIdx.y");
  }

  sch->ReverseComputeInline(state->tensor_core_reindex_store);
  auto loops = sch->GetLoops(cache_write);
  auto blockized_store = sch->Blockize(loops[loops.size() - 2]);
  sch->Annotate(blockized_store, tir::attr::meta_schedule_auto_tensorize,
                state->intrin_group.store_intrin);

  Array<LoopRV> buffer_loops = sch->GetLoops(state->write_reuse[0]);
  ICHECK_GT(buffer_loops.size(), 5);
  sch->Fuse(Array<LoopRV>{buffer_loops.end() - 5,  // The src shmem is always 2D
                          buffer_loops.end()});
  AnnotateCooperativeFetching(&sch, state->write_reuse[0]);
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

  f_tensorize_load(0, "wmma.matrix_a", state->intrin_group.load_a_intrin);
  f_tensorize_load(1, "wmma.matrix_b", state->intrin_group.load_b_intrin);
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
      TVM_PY_LOG(WARNING, logger) << "StorageAlign is not applied for data type " << dtype
                                  << ", shared memory accesses might be inefficient.";
    }
  }
  return {state};
}

std::vector<State> MultiLevelTilingTensorCoreNode::AddSoftwarePipeline(
    TensorCoreState state) const {
  if (!use_software_pipeline) {
    return {state};
  }
  // The current config is not suitable for software pipelining.
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

  // Add local stage and double buffering
  for (int i = 0; i < 2; ++i) {
    const tir::BlockRV cache_read = state->read_reuse.at(i);
    sch->Annotate(cache_read, tir::attr::manifest_shared_memory_local_stage, Integer(1));
    sch->Annotate(cache_read, tir::attr::double_buffer_scope, Integer(0));
  }

  // Add annotations of software pipeline
  //
  // Before pipelining, the original loop can be expressed as the pseudo code below:
  //
  // for k0 in [0, K0):
  //   load tile k0 to registers
  //   load tile k0 from registers to shared memory
  //
  //   for k1 in [0, K1):
  //     load fragment k1 of tile k0
  //     compute matmul with fragment k1
  //

  // Inner software pipeline: Prefetch to tensor core fragment by one iteration
  // The following annotation for the inner loop is equivalent the pesudo code below:
  //
  // Pipelined inner loop:
  //
  // prologue:
  //   load fragment 0
  // body:
  //   for k1 in [0, K1 - 1):
  //     load fragment k1 + 1
  //     compute matmul with fragment k1
  // epilogue:
  //   compute matmul with fragment K1 - 1
  //
  sch->Annotate(state->tiles[r_indices_[1]].back(), tir::attr::software_pipeline_stage,
                Array<Integer>{0, 0, 1});
  sch->Annotate(state->tiles[r_indices_[1]].back(), tir::attr::software_pipeline_order,
                Array<Integer>{0, 1, 2});
  // Outer software pipeline: Interleave the outer loop with the (pipelined) inner loop.
  // The prefetching stage of the inner pipeline is executed by one iteration in the outer loop.
  // The following annotation for the outer loop is equivalent the pesudo code below:
  //
  // Pipelined outer loop with nested inner pipeline:
  //
  // prologue:
  //   load tile 0 to registers
  //   load tile 0 from registers to shared memory
  //
  //   // prologue of the inner pipeline
  //   load fragment 0 of tile 0
  //
  // body:
  //   for k0 in [0, K0 - 1):
  //     load tile k0 + 1 to registers
  //
  //     // body of the inner pipeline
  //     for k1 in [0, K1 - 1):
  //       load fragment k1 + 1 of tile k0
  //       compute matmul with fragment k1 of tile k0
  //
  //     load tile k0 + 1 from registers to shared memory
  //
  //     // prologue of the inner pipeline
  //     load fragment 0 of tile k0 + 1
  //
  //     // epilogue of the inner pipeline
  //     compute matmul with fragment K1 - 1 of tile k0
  //
  // epilogue:
  //
  //   // body of the inner pipeline
  //   for k1 in [0, K1 - 1):
  //     load fragment k1 + 1 of tile K0 - 1
  //     compute matmul with fragment k1 of tile K0 - 1
  //
  //   // epilogue of the inner pipeline
  //   compute matmul with fragment K1 - 1 of tile K0 - 1
  //
  sch->Annotate(state->tiles[r_indices_[0]].back(), tir::attr::software_pipeline_stage,
                Array<Integer>{0, 0, 0, 0, 0, 1, 1});
  sch->Annotate(state->tiles[r_indices_[0]].back(), tir::attr::software_pipeline_order,
                Array<Integer>{0, 3, 1, 4, 5, 2, 6});

  return {state};
}

Optional<LoopRV> MultiLevelTilingTensorCoreNode::TransformWithTensorIntrin(
    TensorCoreStateNode* state, const String& intrin_name) const {
  BlockRV block_rv = state->block_rv;
  const tir::AutoTensorizeMappingInfo& mapping_info = state->mapping_info;
  tir::StmtSRef block_sref = state->sch->GetSRef(state->block_rv);

  // Add reindex stages
  const tir::BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
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
  ICHECK_EQ(mapping_info->mappings.size(), 1U);  // assume only one mapping is present
  const tir::IndexMap& index_map = mapping_info->mappings[0];

  // Find the correspondence between block iters and the iters in the index map.
  std::unordered_map<tir::Var, tir::Var, ObjectPtrHash, ObjectPtrEqual> lhs_to_index_map_src;
  std::unordered_map<tir::Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> rhs_to_index_map_tgt;
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> unmapped_index_map_src;
  ICHECK_EQ(mapping_info->lhs_iters.size(), index_map->initial_indices.size());
  for (int i = 0; i < static_cast<int>(mapping_info->lhs_iters.size()); ++i) {
    lhs_to_index_map_src[mapping_info->lhs_iters[i]->var] = index_map->initial_indices[i];
  }
  // The number of result iters in the index map is equal or more than the number of rhs (the
  // tensor intrin) iters. When there are extra iters, these iters represent unmapped iters from
  // the lhs. They will be skipped during pattern matching for tensorization. An example of such
  // case is batch matmul, the batch dimension is kept after layout transformations and it will be
  // kept as a outer loop after tensorization.
  int offset = static_cast<int>(index_map->final_indices.size()) -
               static_cast<int>(mapping_info->rhs_iters.size());
  ICHECK_GE(offset, 0);
  for (int i = 0; i < offset; ++i) {
    const tir::VarNode* var_ptr = index_map->final_indices[i].as<tir::VarNode>();
    ICHECK(var_ptr != nullptr);
    unmapped_index_map_src.insert(GetRef<tir::Var>(var_ptr));
  }
  for (int i = offset; i < static_cast<int>(index_map->final_indices.size()); ++i) {
    rhs_to_index_map_tgt[mapping_info->rhs_iters[i - offset]->var] = index_map->final_indices[i];
  }

  auto f_get_sub_index_map = [&](const tir::Buffer& lhs_buffer, const tir::Region& lhs_region) {
    std::vector<tir::Var> sub_index_map_src;
    std::vector<PrimExpr> sub_index_map_tgt;
    const tir::Buffer& rhs_buffer = mapping_info->lhs_buffer_map[lhs_buffer];
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
    for (size_t i = 0; i < mapping_info->rhs_buffer_indices[rhs_buffer].size(); ++i) {
      const tir::VarNode* var = mapping_info->rhs_buffer_indices[rhs_buffer][i].as<tir::VarNode>();
      ICHECK(var != nullptr);
      sub_index_map_tgt.push_back(rhs_to_index_map_tgt[GetRef<tir::Var>(var)]);
    }
    return tir::IndexMap(sub_index_map_src, sub_index_map_tgt);
  };

  std::unordered_set<tir::Buffer, ObjectPtrHash, ObjectPtrEqual> visited_buffers;

  Map<tir::Buffer, tir::IndexMap> buffer_sub_index_map;  // cache of the sub index map associated
                                                         // with each buffer

  auto f_transform_buffer_layout = [&](tir::BufferIndexType index_type, int buffer_index) {
    const tir::Buffer& lhs_buffer = tir::GetNthAccessBuffer(
        state->sch->state(), block_before_reindex, buffer_index, index_type);
    if (visited_buffers.count(lhs_buffer)) {
      return;
    }
    visited_buffers.insert(lhs_buffer);
    // Refresh block pointer (block sref is not invalidated)
    block = TVM_SREF_TO_BLOCK(block_sref);
    const tir::BufferRegion& reindexed_buffer_region = tir::GetNthAccessBufferRegion(
        state->sch->state(), GetRef<tir::Block>(block), buffer_index, index_type);
    auto sub_index_map = f_get_sub_index_map(lhs_buffer, reindexed_buffer_region->region);
    buffer_sub_index_map.Set(lhs_buffer, sub_index_map);
    state->sch->TransformLayout(state->block_rv, buffer_index, index_type, sub_index_map,
                                /*pad_value=*/NullOpt, /*assume_injective_transform=*/true);
  };

  for (int i = 0, n = block_before_reindex->reads.size(); i < n; ++i) {
    f_transform_buffer_layout(tir::BufferIndexType::kRead, i);
  }
  for (int i = 0, n = block_before_reindex->writes.size(); i < n; ++i) {
    f_transform_buffer_layout(tir::BufferIndexType::kWrite, i);
  }

  // Transform the layout of current block and reindex blocks
  auto f_transform_reindex_block_layout = [&](const BlockRV& block_rv,
                                              tir::BufferIndexType buffer_type) {
    tir::Buffer buffer =
        tir::GetNthAccessBuffer(state->sch->state(), state->sch->Get(block_rv), 0, buffer_type);
    const auto& sub_index_map = buffer_sub_index_map.at(buffer);
    state->sch->TransformBlockLayout(block_rv, sub_index_map);
  };
  f_transform_reindex_block_layout(state->tensor_core_reindex_store, tir::BufferIndexType::kWrite);
  f_transform_reindex_block_layout(state->tensor_core_reindex_A, tir::BufferIndexType::kRead);
  f_transform_reindex_block_layout(state->tensor_core_reindex_B, tir::BufferIndexType::kRead);
  state->sch->TransformBlockLayout(state->block_rv, index_map);
  return tir::TileWithTensorIntrin(state->sch, state->block_rv, intrin_name,
                                   /*allow_padding=*/true);
}

inline std::vector<State> MultiLevelTilingTensorCoreNode::TransformForTensorization(
    TensorCoreState state) const {
  // Do reindex and layout transformations.
  Optional<LoopRV> transformed_loop_rv =
      TransformWithTensorIntrin(state.operator->(), state->intrin_group.compute_intrin);
  if (!transformed_loop_rv.defined()) {
    // The workload can't be tensorized.
    return {};
  }

  // Do blockize
  state->block_rv = state->sch->Blockize(transformed_loop_rv.value());

  // Add annotations for post processors.
  state->sch->Annotate(state->block_rv, tir::attr::meta_schedule_auto_tensorize,
                       state->intrin_group.compute_intrin);
  state->sch->Annotate(state->block_rv, tir::attr::meta_schedule_auto_tensorize_init,
                       state->intrin_group.init_intrin);
  state->sch->Annotate(state->block_rv, tir::attr::warp_execution, Integer(1));
  return {std::move(state)};
}

ScheduleRule ScheduleRule::MultiLevelTilingTensorCore(
    Array<Map<String, String>> intrin_groups, String structure, Optional<Array<String>> tile_binds,
    Optional<Integer> max_innermost_factor, Optional<Array<Integer>> vector_load_lens,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write,
    bool use_software_pipeline) {
  if (tile_binds.defined()) {
    for (const String& tile_bind : tile_binds.value()) {
      CHECK_NE(tile_bind, "threadIdx.x") << "Cannot bind to threadIdx.x when using tensor core.";
    }
  }
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingTensorCoreNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);

  CHECK(node->reuse_write_.req == ReuseType::kMustReuse &&
        runtime::StorageScope::Create(node->reuse_write_.scope).rank ==
            runtime::StorageRank::kShared)
      << "ValueError: Shared memory write reuse must be enabled for MultiLevelTilingTensorCore.";

  node->intrin_groups.reserve(intrin_groups.size());
  for (const auto& intrin_group_config : intrin_groups) {
    node->intrin_groups.emplace_back(TensorCoreIntrinGroup::FromConfig(intrin_group_config));
  }
  node->use_software_pipeline = use_software_pipeline;
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingTensorCoreNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingTensorCore")
    .set_body_typed(ScheduleRule::MultiLevelTilingTensorCore);

}  // namespace meta_schedule
}  // namespace tvm
