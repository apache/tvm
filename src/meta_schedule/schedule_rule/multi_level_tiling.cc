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
#include "./multi_level_tiling.h"

#include <tvm/meta_schedule/schedule_rule.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace tir {
/*!
 * \brief Get the buffer dimensions for all the read buffers of a block, but marks the reduction
 * buffers' dimensions as -1
 * \param block_sref The block to be processed
 * \return The buffer dimensions for all the read buffers of a block, except for reduction buffers
 * \note The method is not designed for generic analysis and relies on assumptions in the scenario
 * of multi-level tiling, so it's intentionally kept inside this file not in the analysis header
 */
std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const BufferNode* write_buffer = block->writes[0]->buffer.get();
  int n = block->reads.size();
  std::vector<int> results(n, -1);
  for (int i = 0; i < n; ++i) {
    const BufferNode* read_buffer = block->reads[i]->buffer.get();
    if (read_buffer != write_buffer) {
      results[i] = read_buffer->shape.size();
    }
  }
  return results;
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

TVM_REGISTER_OBJECT_TYPE(StateNode);

State::State(tir::Schedule sch, tir::BlockRV block_rv, Array<Array<tir::LoopRV>> tiles) {
  ObjectPtr<StateNode> node = make_object<StateNode>();
  node->sch = std::move(sch);
  node->block_rv = std::move(block_rv);
  node->tiles = std::move(tiles);
  data_ = std::move(node);
}

State StateNode::Copy() const {
  ObjectPtr<StateNode> node = make_object<StateNode>(*this);
  node->sch = sch->Copy();
  return State(node);
}

// Do nothing; Inherited from ScheduleRuleNode
void MultiLevelTilingNode::InitializeWithTuneContext(const TuneContext& context) {
  if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("max_threads_per_block")) {
    this->max_threads_per_block_ = v.value()->value;
    if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("thread_warp_size")) {
      this->thread_warp_size_ = v.value()->value;
    } else {
      TVM_PY_LOG(INFO, context->logger) << "'thread_warp_size' is not defined in the target";
    }
  }
  logger = context->logger;
}

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingNode::Apply(const Schedule& sch, const BlockRV& block_rv) {
  if (!NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
    return {sch};
  }
  sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);

  Array<Schedule> results;
  for (auto&& state : ApplySubRules({State(sch, block_rv)})) {
    results.push_back(std::move(state->sch));
  }
  return results;
}

// Inherited from ScheduleRuleNode
ScheduleRule MultiLevelTilingNode::Clone() const {
  ObjectPtr<MultiLevelTilingNode> n = make_object<MultiLevelTilingNode>(*this);
  return ScheduleRule(n);
}

std::vector<State> MultiLevelTilingNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(std::move(state)); });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(std::move(state)); });
  return states;
}

std::vector<State> MultiLevelTilingNode::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  std::vector<int> levels = config.levels;
  ReuseType req = config.req;
  if (Optional<Array<Integer>> ann = tir::GetAnn<Array<Integer>>(
          state->sch->GetSRef(state->block_rv), "meta_schedule.write_cache_level")) {
    req = ReuseType::kMustReuse;
    levels.clear();
    std::transform(ann.value().begin(), ann.value().end(), std::back_inserter(levels),
                   [](auto&& v) { return v.IntValue(); });
  }
  std::vector<State> results;
  if (req == ReuseType::kMayReuse) {
    // Case 1. If the write cache is already there, we don't need to add another.
    Array<BlockRV> consumer_rvs = state->sch->GetConsumers(state->block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state->sch->GetSRef(consumer_rvs[0]))) {
      for (int level : levels) {
        State new_state = state->Copy();
        const LoopRV& loop_rv = new_state->tiles[level - 1].back();
        new_state->sch->ReverseComputeAt(consumer_rvs[0], loop_rv, true);
        results.push_back(std::move(new_state));
      }
      state->write_reuse.emplace(0, consumer_rvs[0]);
      results.push_back(state);
      return results;
    } else {
      // Case 2. No write cache is added
      State new_state = state->Copy();
      results.emplace_back(std::move(new_state));
    }
  }

  // Case 3. Add one write cache
  BlockRV write_cache =
      state->sch->CacheWrite(/*block_rv=*/state->block_rv, /*read_buffer_index=*/0,
                             /*storage_scope=*/config.scope);
  state->write_reuse.emplace(0, write_cache);
  for (int level : levels) {
    State new_state = state->Copy();
    const LoopRV& loop_rv = new_state->tiles[level - 1].back();
    new_state->sch->ReverseComputeAt(write_cache, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  return results;
}

Array<tir::LoopRV> MultiLevelTilingNode::SplitLoop(const Schedule& sch, BlockRV block, LoopRV loop,
                                                   int n_tiles) const {
  Array<tir::ExprRV> factors = sch->SamplePerfectTile(
      /*loop=*/loop,
      /*n=*/n_tiles,
      /*max_innermost_factor=*/max_innermost_factor);
  Array<tir::LoopRV> splits = sch->Split(/*loop=*/loop,
                                         /*factors=*/{factors.begin(), factors.end()});
  return splits;
}

std::vector<State> MultiLevelTilingNode::TileLoopNest(State state) const {
  Schedule& sch = state->sch;
  const BlockRV& block_rv = state->block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state->block_rv));
  ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  int64_t spatial_loop_product = 1;
  std::vector<Array<LoopRV>> tiles(s_indices_.size() + r_indices_.size());
  for (int i = 0, n = loops.size(); i < n; ++i) {
    LoopRV loop = loops[i];
    const std::vector<int>* idx = nullptr;

    if (iter_types[i] == IterVarType::kDataPar) {
      idx = &s_indices_;
      if (spatial_loop_product != -1) {
        if (const int64_t* extent = tir::GetLoopIntExtent(sch->Get(loop).get())) {
          spatial_loop_product *= *extent;
        } else {
          spatial_loop_product = -1;
        }
      }
    } else if (iter_types[i] == IterVarType::kCommReduce) {
      idx = &r_indices_;
    } else {
      continue;
    }

    const int n_tiles = idx->size();

    if (n_tiles == 1) {
      tiles[idx->at(0)].push_back(loop);
    } else {
      auto splits = SplitLoop(sch, block_rv, loop, n_tiles);

      // Put every tile to its slot
      for (int j = 0; j < n_tiles; ++j) {
        tiles[idx->at(j)].push_back(splits[j]);
      }
    }
  }
  // Step 3. Reorder to organize the tiles
  sch->Reorder(support::ConcatArrayList<LoopRV>(tiles.begin(), tiles.end()));
  // Step 4. Bind the tiles to threads
  int n_binds = std::min(tile_binds.size(), tiles.size());
  for (int i = 0; i < n_binds; ++i) {
    LoopRV fused = sch->Fuse(tiles[i]);
    sch->Bind(fused, tile_binds[i]);
    tiles[i] = {fused};
  }
  state->tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
  if (this->thread_warp_size_ != -1) {
    int64_t low_inclusive = 1;
    int64_t high_inclusive = this->max_threads_per_block_;
    if (spatial_loop_product > 2 * this->thread_warp_size_) {
      low_inclusive = this->thread_warp_size_;
    }
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_low_inclusive,
                  Integer(low_inclusive));
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_high_inclusive,
                  Integer(high_inclusive));
  }
  return {state};
}

std::vector<State> MultiLevelTilingNode::AddReadReuse(State state) const {
  const ReuseConfig& config = this->reuse_read_;
  if (config.req == ReuseType::kNoReuse) {
    return {std::move(state)};
  }
  ICHECK(config.req != ReuseType::kMayReuse);
  const BlockRV& block_rv = state->block_rv;
  std::vector<State> results;
  results.reserve(config.levels.size());
  for (int level : config.levels) {
    State new_state = state->Copy();
    Schedule& sch = new_state->sch;
    const LoopRV& loop_rv = state->tiles[level - 1].back();
    // Enumerate all buffers that are read but not written
    std::vector<int> read_buffer_ndims = tir::GetReadBufferNDims(sch->GetSRef(block_rv));
    for (int i = 0, n_reads = read_buffer_ndims.size(); i < n_reads; ++i) {
      int buffer_ndim = read_buffer_ndims[i];
      if (buffer_ndim == -1) {
        continue;
      }
      // Do cache_read
      BlockRV cache_read_block = sch->CacheRead(block_rv, i, config.scope, {block_rv});
      // Insert cache_read block to the proper place
      sch->ComputeAt(cache_read_block, loop_rv, true);
      // Fuse the iterators of the cache_read
      Array<LoopRV> buffer_loops = sch->GetLoops(cache_read_block);
      LoopRV fused = sch->Fuse(Array<LoopRV>{buffer_loops.end() - buffer_ndim,  //
                                             buffer_loops.end()});
      AnnotateCooperativeFetching(&sch, cache_read_block);
      new_state->read_reuse.emplace(i, cache_read_block);
    }
    results.push_back(std::move(new_state));
  }
  return results;
}

void MultiLevelTilingNode::AnnotateCooperativeFetching(Schedule* sch,
                                                       const tir::BlockRV& block) const {
  // Filter out invalid vector lanes according to the data type.
  const tir::BlockNode* block_node = (*sch)->GetSRef(block)->StmtAs<tir::BlockNode>();
  ICHECK_EQ(block_node->writes.size(), 1);
  const runtime::DataType dtype = block_node->writes[0]->buffer->dtype;
  std::function<bool(int)> f_filter = nullptr;
  if (dtype == runtime::DataType::Float(32)) {
    f_filter = [&](int vector_len) { return vector_len <= 4; };
  } else if (dtype == runtime::DataType::Float(16)) {
    f_filter = [&](int vector_len) {
      return (vector_len == 1 || vector_len % 2 == 0) && vector_len <= 8;
    };
  } else if (dtype == runtime::DataType::Int(8)) {
    f_filter = [&](int vector_len) { return vector_len <= 16; };
  }
  std::vector<int> valid_vector_lens;
  valid_vector_lens.reserve(vector_load_lens.size());
  if (f_filter != nullptr) {
    std::copy_if(vector_load_lens.begin(), vector_load_lens.end(),
                 std::back_inserter(valid_vector_lens), f_filter);
  } else {
    valid_vector_lens = vector_load_lens;
  }

  if (!valid_vector_lens.empty()) {
    int n = valid_vector_lens.size();
    double prob = 1.0 / n;
    tir::ExprRV vector_load_len =
        (*sch)->SampleCategorical(support::AsArray<int, Integer>(valid_vector_lens),
                                  Array<FloatImm>(n, FloatImm(DataType::Float(64), prob)));
    (*sch)->Annotate(block, tir::attr::meta_schedule_cooperative_fetch, vector_load_len);
  }
}

// Constructor

ScheduleRule ScheduleRule::MultiLevelTiling(String structure, Optional<Array<String>> tile_binds,
                                            Optional<Integer> max_innermost_factor,
                                            Optional<Array<Integer>> vector_load_lens,
                                            Optional<Map<String, ObjectRef>> reuse_read,
                                            Optional<Map<String, ObjectRef>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTiling")
    .set_body_typed(ScheduleRule::MultiLevelTiling);

}  // namespace meta_schedule
}  // namespace tvm
