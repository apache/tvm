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
 * \file auto_scheduler/search_policy/sketch_policy_rules.cc
 * \brief Rules defined to generate the sketches and initial sampled states in SketchPolicy.
 */

#include "sketch_policy_rules.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "sketch_policy.h"

namespace tvm {
namespace auto_scheduler {

/********** Sketch Generation Rule **********/
/********** RuleSkipStage **********/

SketchGenerationRule::ConditionKind RuleSkipStage::MeetCondition(const SketchPolicyNode& policy,
                                                                 const State& state, int stage_id) {
  // This rule should be the last rule, always return true to decrease the stage index count
  return ConditionKind::kApply;
}

std::vector<std::pair<State, int>> RuleSkipStage::Apply(const SketchPolicyNode& policy,
                                                        const State& state, int stage_id) const {
  return {std::make_pair(state, stage_id - 1)};
}

/********** RuleAlwaysInline **********/

SketchGenerationRule::ConditionKind RuleAlwaysInline::MeetCondition(const SketchPolicyNode& policy,
                                                                    const State& state,
                                                                    int stage_id) {
  const Stage& stage = state->stages[stage_id];
  // Check the inline limitation of TE first
  if (stage->op_type == StageKind::kPlaceholder ||
      IsOutputOp(policy.search_task, state, stage_id) || HasReduceIter(stage)) {
    return ConditionKind::kSkip;
  }

  // TODO(jcf94): Greedily inline all inlinable ops on GPU when introducing GPU search policy.
  return IsStrictlyInlineable(policy.search_task, state, stage_id)
             ? ConditionKind::kApplyAndSkipRest
             : ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleAlwaysInline::Apply(const SketchPolicyNode& policy,
                                                           const State& state, int stage_id) const {
  State tmp_s = state;
  tmp_s.compute_inline(stage_id);
  return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** RuleMultiLevelTiling **********/

SketchGenerationRule::ConditionKind RuleMultiLevelTiling::MeetCondition(
    const SketchPolicyNode& policy, const State& state, int stage_id) {
  return NeedsMultilevelTiling(policy.search_task, state, stage_id)
             ? ConditionKind::kApplyAndSkipRest
             : ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleMultiLevelTiling::Apply(const SketchPolicyNode& policy,
                                                               const State& state,
                                                               int stage_id) const {
  // TODO(jcf94): Add support for GPU structure when introducing GPU search policy.
  const std::string& multi_level_tiling_structure =
      GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::cpu_structure);
  State tmp_s = DoMultiLevelTiling(state, stage_id, multi_level_tiling_structure);
  return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** RuleMultiLevelTilingWithFusion **********/

SketchGenerationRule::ConditionKind RuleMultiLevelTilingWithFusion::MeetCondition(
    const SketchPolicyNode& policy, const State& state, int stage_id) {
  if (NeedsMultilevelTiling(policy.search_task, state, stage_id) &&
      HasSingleElementwiseMatchedConsumer(policy.search_task, state, stage_id, &target_stage_id)) {
    // Always do fusion for stage with cache_write
    // TODO(jcf94): Always do fusion on GPU when introducing GPU search policy.
    return HasCacheWriteStage(state, stage_id) ? ConditionKind::kApplyAndSkipRest
                                               : ConditionKind::kApply;
  }
  return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleMultiLevelTilingWithFusion::Apply(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  // TODO(jcf94): Add support for GPU structure when introducing GPU search policy.
  const std::string& multi_level_tiling_structure =
      GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::cpu_structure);
  std::vector<int> spatial_split_step_ids;
  State base_state =
      DoMultiLevelTiling(state, stage_id, multi_level_tiling_structure, &spatial_split_step_ids);

  std::vector<std::pair<State, int>> ret;
  // TODO(jcf94): Add follow_tiling_levels for GPU when introducing GPU search policy.
  std::vector<int> follow_tiling_levels{1, 2};
  for (int level : follow_tiling_levels) {
    if (tolower(multi_level_tiling_structure[level - 1]) != 's') {
      continue;
    }
    State tmp_s = base_state;
    tmp_s = FollowTiling(tmp_s, target_stage_id, spatial_split_step_ids, level);
    const Iterator& target_iter =
        tmp_s->stages[target_stage_id]->iters[level * spatial_split_step_ids.size() - 1];
    tmp_s.compute_at(stage_id, target_stage_id, target_iter);
    ret.emplace_back(std::move(tmp_s), stage_id - 1);
  }

  return ret;
}

/********** RuleAddCacheWrite **********/

SketchGenerationRule::ConditionKind RuleAddCacheWrite::MeetCondition(const SketchPolicyNode& policy,
                                                                     const State& state,
                                                                     int stage_id) {
  // Add cache write if a stage needs multi-level tiling, but does not have a element-wise
  // matched consumer
  if (NeedsMultilevelTiling(policy.search_task, state, stage_id) &&
      !HasSingleElementwiseMatchedConsumer(policy.search_task, state, stage_id)) {
    // An apply and skip rule will be handled in RuleMultiLevelTilingWithFusion
    // TODO(jcf94): Always do cache_write on GPU when introducing GPU search policy.
    return ConditionKind::kApply;
  }

  return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleAddCacheWrite::Apply(const SketchPolicyNode& policy,
                                                            const State& state,
                                                            int stage_id) const {
  State tmp_s = state;
  tmp_s.cache_write(stage_id, "local", policy.search_task->compute_dag);
  return {std::make_pair(std::move(tmp_s), stage_id)};
}

/********** RuleAddRfactor **********/

SketchGenerationRule::ConditionKind RuleAddRfactor::MeetCondition(const SketchPolicyNode& policy,
                                                                  const State& state,
                                                                  int stage_id) {
  return (NeedsRfactor(policy.search_task, state, stage_id) && !HasCacheWriteStage(state, stage_id))
             ? ConditionKind::kApply
             : ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleAddRfactor::Apply(const SketchPolicyNode& policy,
                                                         const State& state, int stage_id) const {
  // Fuse all reduction iters
  Array<Iterator> space_iters, reduce_iters;
  Iterator fused_reduce_iter;
  State base_state =
      FuseAllReductionIterators(state, stage_id, &fused_reduce_iter, &space_iters, &reduce_iters);

  // TODO(merrymercy): We can do more analysis here to generate less and more efficient sketches.
  // In some cases, we only need rfactor for more parallel
  // In some cases, we only need rfactor for vectorization.
  // Now we will generate two versions and let the search figure out the bette one.

  // Split reduction iters
  const auto& split_res = base_state.split(stage_id, fused_reduce_iter, {Integer(1)});
  int factor_axis_id = static_cast<int>(space_iters.size());
  std::vector<std::pair<State, int>> ret;
  for (const auto& split_iter : split_res) {
    State tmp_s = base_state;
    int rstage_id =
        tmp_s.rfactor(stage_id, split_iter, factor_axis_id, policy.search_task->compute_dag);

    // reorder the space iterator to innermost for vectorization
    if (split_iter == split_res[1]) {
      Array<Iterator> new_order;
      for (size_t i = 0; i < tmp_s->stages[rstage_id]->iters.size(); ++i) {
        if (i != space_iters.size()) {
          new_order.push_back(tmp_s->stages[rstage_id]->iters[i]);
        }
      }
      new_order.push_back(tmp_s->stages[rstage_id]->iters[space_iters.size()]);
      tmp_s.reorder(rstage_id, new_order);
    }

    ret.emplace_back(std::move(tmp_s), rstage_id - 1);
  }

  return ret;
}

/********** RuleSimplifyComputeWithConstTensor **********/

SketchGenerationRule::ConditionKind RuleSimplifyComputeWithConstTensor::MeetCondition(
    const SketchPolicyNode& policy, const State& state, int stage_id) {
  return state->stages[stage_id]->op->attrs.count(SearchPolicyKey::simplify_const_tensor_indices)
             ? ConditionKind::kApplyAndSkipRest
             : ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleSimplifyComputeWithConstTensor::Apply(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  std::set<std::string> const_tensor_indices = GetIterNameSetParam(
      state->stages[stage_id]->op->attrs, SearchPolicyKey::simplify_const_tensor_indices);

  State tmp_s = state;
  Array<Array<Iterator>> tiled_outer_iters;
  Array<Iterator> unrolled_inner_iters;

  // Currently set to 2
  size_t tile_level = 2;

  for (const auto& iter : state->stages[stage_id]->iters) {
    if (const_tensor_indices.count(iter->name)) {
      // unroll indices of const tensors
      unrolled_inner_iters.push_back(tmp_s.unroll(stage_id, iter));
    } else {
      // tile other space indices
      CHECK(iter->iter_kind == IteratorKind::kSpatial);
      tiled_outer_iters.push_back(
          tmp_s.split(stage_id, iter, Array<Optional<Integer>>(tile_level - 1, NullOpt)));
    }
  }

  // reorder them
  Array<Iterator> new_order;
  for (size_t i = 0; i < tile_level; ++i) {
    for (size_t j = 0; j < tiled_outer_iters.size(); ++j) {
      new_order.push_back(tiled_outer_iters[j][i]);
    }
  }
  new_order.insert(new_order.end(), unrolled_inner_iters.begin(), unrolled_inner_iters.end());
  tmp_s.reorder(stage_id, new_order);

  return {std::make_pair(tmp_s, stage_id - 1)};
}

/********** Init Population **********/

InitPopulationRule::ResultKind InitFillTileSize::Apply(SketchPolicyNode* policy,
                                                       State* state) const {
  StateNode* pstate = state->CopyOnWrite();
  // Scan the transformation history and randomly fill tiles size for all SplitStep
  for (size_t step_id = 0; step_id < (*state)->transform_steps.size(); ++step_id) {
    if (auto ps = (*state)->transform_steps[step_id].as<SplitStepNode>()) {
      bool all_defined = true;
      for (const auto& len : ps->lengths) {
        if (!len) {
          all_defined = false;
          break;
        }
      }
      if (all_defined) {
        continue;
      }

      CHECK(ps->extent);
      int extent = GetIntImm(ps->extent.value());
      const auto& candidate_lens = policy->split_memo.GetFactorizationSchemes(
          extent, ps->lengths.size(),
          GetIntParam(policy->params, SketchParamKey::max_innermost_split_factor));
      const auto& candidate_lengths = candidate_lens[(policy->rand_gen)() % candidate_lens.size()];

      pstate->transform_steps.Set(
          step_id,
          SplitStep(ps->stage_id, ps->iter_id, ps->extent,
                    Array<Optional<Integer>>(candidate_lengths.begin(), candidate_lengths.end()),
                    ps->inner_to_outer));
    }
  }
  pstate->concrete = true;

  return ResultKind::kValid;
}

InitPopulationRule::ResultKind InitChangeComputeLocation::Apply(SketchPolicyNode* policy,
                                                                State* state) const {
  if (GetIntParam(policy->params, SketchParamKey::disable_change_compute_location)) {
    return ResultKind::kValid;
  }

  for (int stage_id = static_cast<int>((*state)->stages.size()) - 1; stage_id >= 0; stage_id--) {
    const Stage& stage = (*state)->stages[stage_id];
    // Skip the inlined stages and placeholders
    if (stage->op_type == StageKind::kPlaceholder || stage->compute_at == ComputeAtKind::kInlined) {
      continue;
    }
    // Skip the tiled stages
    if (IsTiled(stage) || NeedsMultilevelTiling(policy->search_task, *state, stage_id)) {
      continue;
    }

    int target_stage_id = GetSingleConsumerId(policy->search_task, *state, stage_id);
    if (target_stage_id < 0) {
      continue;
    }
    const Stage& target_stage = (*state)->stages[target_stage_id];

    std::vector<std::pair<int, int>> candidates;
    bool target_compute_at_other = target_stage->compute_at == ComputeAtKind::kIter;
    bool target_is_tiled = IsTiled(target_stage);

    bool visited_reduce = false;
    // enumerate compute_at location at target_stage
    // TODO(merrymercy): More analysis here to make smarter choices
    for (size_t i = 0; i < target_stage->iters.size(); ++i) {
      const Iterator& target_iter = target_stage->iters[i];
      if (target_iter->iter_kind == IteratorKind::kReduction) {
        visited_reduce = true;
        if (!target_is_tiled) {  // Do not go into reduce iter
          break;
        }
      } else if (target_iter->iter_kind == IteratorKind::kSpatial) {
        if (visited_reduce) {  // Do not go into inner tile
          break;
        }
      }

      if (target_iter->annotation == IteratorAnnotation::kUnroll) {
        // Do not go into the unroll region of const tensor indices
        break;
      }

      if (GetExtent(target_iter) == 1) {
        // Skip iterators with length of 1
        continue;
      }
      if (target_compute_at_other && target_iter->iter_kind == IteratorKind::kSpatial &&
          StrEndsWith(target_iter->name, ".0")) {
        // Skip the first level iterators if target stage compute_at another stage
        // In this case, the lengths of first level iterators are always one
        continue;
      }
      candidates.emplace_back(target_stage_id, i);

      if ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(target_stage_id, i))) {
        break;
      }
    }

    // if the target_stage is already compute_at another stage X, try also compute_at X
    // We call stage X as `target_target_stage`
    if (target_compute_at_other) {
      int target_target_stage_id;
      target_target_stage_id = (*state)->attach_map->stage_to_attach_iter.at(target_stage_id).first;
      const Stage& target_target_stage = (*state)->stages[target_target_stage_id];

      for (size_t i = 0; i < target_target_stage->iters.size(); ++i) {
        const Iterator& target_target_iter = target_target_stage->iters[i];
        if (target_target_iter->iter_kind == IteratorKind::kReduction ||
            (*state)->attach_map->iter_to_attached_stages.count(
                std::make_pair(target_target_stage_id, i))) {
          break;
        }

        if (target_target_iter->annotation == IteratorAnnotation::kUnroll) {
          // Do not go into the unroll region of const tensor indices
          break;
        }

        if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
          continue;
        }

        candidates.emplace_back(target_target_stage_id, i);
      }
    }

    int choice = (policy->rand_gen)() % (candidates.size() + 2);

    if (choice == 0) {
      if (!HasReduceIter(stage)) {
        const auto& stage_to_attach_iter = (*state)->attach_map->stage_to_attach_iter;
        if (stage_to_attach_iter.find(stage_id) != stage_to_attach_iter.end()) {
          state->compute_inline(stage_id);
        }
      }
    } else if (choice == 1) {
      state->compute_root(stage_id);
    } else {
      choice = choice - 2;
      const Stage& stage = (*state)->stages[candidates[choice].first];
      state->compute_at(stage_id, candidates[choice].first,
                        stage->iters[candidates[choice].second]);
    }
  }

  *state = policy->search_task->compute_dag.InferBound(*state);
  return ResultKind::kValid;
}

InitPopulationRule::ResultKind InitParallel::Apply(SketchPolicyNode* policy, State* state) const {
  std::function<void(const SketchPolicyNode&, State*, int stage_id, int iter_offset)>
      annotate_parallel;
  annotate_parallel = [&annotate_parallel](const SketchPolicyNode& policy, State* state,
                                           int stage_id, int iter_offset) {
    const Stage& stage = (*state)->stages[stage_id];

    Array<Iterator> to_fuse;
    int64_t parallel_degree = 1;

    // Try to fuse and parallel the outermost n iterators
    // Stop if we meet reduce iterator or we have enough parallel degree
    size_t iter_id = iter_offset;
    for (; iter_id < stage->iters.size(); ++iter_id) {
      const Iterator& it = stage->iters[iter_id];
      if (it->iter_kind == IteratorKind::kReduction ||
          it->annotation != IteratorAnnotation::kNone) {
        break;
      }
      to_fuse.push_back(it);
      parallel_degree *= GetExtent(it);

      if (parallel_degree > policy.search_task->hardware_params->num_cores * 16) {
        break;
      }

      if ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id))) {
        break;
      }
    }

    if (parallel_degree == 1) {
      auto res =
          (*state)->attach_map->iter_to_attached_stages.find(std::make_pair(stage_id, iter_id));
      if (res != (*state)->attach_map->iter_to_attached_stages.end()) {
        for (int attached_stage_id : res->second) {
          annotate_parallel(policy, state, attached_stage_id, 0);
        }
        annotate_parallel(policy, state, stage_id, iter_id + 1);
      }
    }

    if (!to_fuse.empty()) {
      if (to_fuse.size() == 1) {
        state->parallel(stage_id, to_fuse[0]);
      } else {
        Iterator fused_iter = state->fuse(stage_id, to_fuse);
        state->parallel(stage_id, fused_iter);
      }
    }
  };

  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];
    if (stage->compute_at != ComputeAtKind::kRoot || stage->op_type == StageKind::kPlaceholder) {
      continue;
    }

    annotate_parallel(*policy, state, stage_id, 0);
  }

  return ResultKind::kValid;
}

InitPopulationRule::ResultKind InitUnroll::Apply(SketchPolicyNode* policy, State* state) const {
  std::vector<int> auto_unroll_configs = {0, 16, 64, 512};
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];
    // Skip the inlined stage and placeholder stage
    if (stage->compute_at == ComputeAtKind::kInlined || stage->op_type == StageKind::kPlaceholder) {
      continue;
    }

    // Handle always_unroll_inner attr
    if (stage->op->attrs.count(SearchPolicyKey::always_unroll_inner)) {
      const auto& to_unroll_name_set =
          GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::always_unroll_inner);

      // Unroll the space iterators and reduce iterators listed in the attrs in the innermost
      // tile
      std::set<std::string> visited_names;
      for (int n = static_cast<int>(stage->iters.size()) - 1; n >= 0; n--) {
        const Iterator& it = stage->iters[n];

        // If we meet two iterators that come from a same original iterator,
        // then we are out of the innermost tile
        size_t size_before = visited_names.size();
        ExtractOriginalIterators(it->name, &visited_names);
        if (size_before == visited_names.size()) {
          break;
        }

        std::set<std::string> name;
        ExtractOriginalIterators(it->name, &name);
        if (name.size() == 1 && to_unroll_name_set.count(*name.begin())) {
          if (it->annotation == IteratorAnnotation::kNone) {
            state->unroll(stage_id, it);
          }
        }
      }
    }

    if (HasReduceIter(stage)) {
      // Use auto unroll for multi level tiled stage
      int value = auto_unroll_configs[(policy->rand_gen)() % auto_unroll_configs.size()];
      state->pragma(stage_id, (*state)->stages[stage_id]->iters[0],
                    std::string("auto_unroll_max_step") + "$" + std::to_string(value));
    }
  }

  return ResultKind::kValid;
}

InitPopulationRule::ResultKind InitVectorization::Apply(SketchPolicyNode* policy,
                                                        State* state) const {
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];
    // Skip the inlined stage and placeholder stage
    if (stage->compute_at == ComputeAtKind::kInlined || stage->op_type == StageKind::kPlaceholder) {
      continue;
    }

    // Try to fuse and vectorize the space iterators in the inner most tile
    int64_t cum_length_prod = 1;

    int num_fusible = 0;
    while (num_fusible < static_cast<int>(stage->iters.size())) {
      int iter_id = static_cast<int>(stage->iters.size()) - 1 - num_fusible;
      // Stop if this iterator has been a compute at attatch point
      if ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id))) {
        break;
      }

      const Iterator& it = stage->iters[iter_id];
      // Stop if we meet a reduce iterator or annotated iterator
      if (it->iter_kind == IteratorKind::kReduction ||
          it->annotation != IteratorAnnotation::kNone) {
        break;
      }

      // Stop if the memory access is not continuous (vectorizable)
      // Note: The check is too hard, so we use heuristic here
      if (IsTiled(stage) && num_fusible != 0) {
        // If the stage is tiled, then the memory access must not be continuous
        // for the innermost two iterators
        break;
      }

      cum_length_prod *= GetExtent(it);
      if (cum_length_prod > GetIntParam(policy->params, SketchParamKey::max_vectorize_size)) {
        break;
      }

      num_fusible++;
    }

    if (num_fusible > 1) {
      // Select a random range to fuse
      num_fusible = 1 + (policy->rand_gen)() % (num_fusible - 1);
    }

    if (num_fusible == 1) {
      state->vectorize(stage_id, stage->iters.back());
    } else if (num_fusible > 1) {
      Array<Iterator> to_fuse(stage->iters.end() + (-num_fusible), stage->iters.end());
      state->vectorize(stage_id, state->fuse(stage_id, to_fuse));
    }
  }

  return ResultKind::kValid;
}

}  // namespace auto_scheduler
}  // namespace tvm
