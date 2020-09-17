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
                                                                 const State& state,
                                                                 int stage_id) const {
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
                                                                    int stage_id) const {
  const Stage& stage = state->stages[stage_id];
  // Check the inline limitation of TE first
  if (stage->op_type == StageKind::kPlaceholder ||
      IsOutputOp(policy.search_task, state, stage_id) || HasReduceIter(stage)) {
    return ConditionKind::kSkip;
  }

  // Always do compute inline if it's strictly inlineable or is in GPU policy
  return IsStrictlyInlineable(policy.search_task, state, stage_id) || IsGPUTask(policy.search_task)
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
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  return NeedsMultilevelTiling(policy.search_task, state, stage_id)
             ? ConditionKind::kApplyAndSkipRest
             : ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleMultiLevelTiling::Apply(const SketchPolicyNode& policy,
                                                               const State& state,
                                                               int stage_id) const {
  const std::string& multi_level_tiling_structure =
      IsGPUTask(policy.search_task)
          ? GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::gpu_structure)
          : GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::cpu_structure);
  State tmp_s = DoMultiLevelTiling(state, stage_id, multi_level_tiling_structure);
  return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** RuleMultiLevelTilingWithFusion **********/

SketchGenerationRule::ConditionKind RuleMultiLevelTilingWithFusion::MeetCondition(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  if (NeedsMultilevelTiling(policy.search_task, state, stage_id) &&
      HasSingleElementwiseMatchedConsumer(policy.search_task, state, stage_id)) {
    // Always do fusion for stage with cache_write or is in GPU policy
    return HasCacheWriteStage(state, stage_id) || IsGPUTask(policy.search_task)
               ? ConditionKind::kApplyAndSkipRest
               : ConditionKind::kApply;
  }
  return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleMultiLevelTilingWithFusion::Apply(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  int target_stage_id;
  CHECK(HasSingleElementwiseMatchedConsumer(policy.search_task, state, stage_id, &target_stage_id));
  const std::string& multi_level_tiling_structure =
      IsGPUTask(policy.search_task)
          ? GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::gpu_structure)
          : GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::cpu_structure);
  std::vector<int> spatial_split_step_ids;
  State base_state =
      DoMultiLevelTiling(state, stage_id, multi_level_tiling_structure, &spatial_split_step_ids);

  std::vector<std::pair<State, int>> ret;
  std::vector<int> follow_tiling_levels =
      IsGPUTask(policy.search_task) ? std::vector<int>{3} : std::vector<int>{1, 2};
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

/********** RuleAddCacheRead **********/

SketchGenerationRule::ConditionKind RuleAddCacheRead::MeetCondition(const SketchPolicyNode& policy,
                                                                    const State& state,
                                                                    int stage_id) const {
  const SearchTask& task = policy.search_task;

  // Don't cache_read a stage if it has multiple consumers
  const std::set<int>& consumers = GetConsumers(task, state, stage_id);
  if (consumers.size() != 1) {
    return ConditionKind::kSkip;
  }

  // Don't cache_read a stage if its consumer does not need multi-level tiling
  int target_stage_id = *consumers.begin();
  if (!NeedsMultilevelTiling(task, state, target_stage_id)) {
    return ConditionKind::kSkip;
  }

  // Don't cache_read a stage if its consumer does cross-thread reduction
  if (HasCrossThreadReduction(state, target_stage_id)) {
    return ConditionKind::kSkip;
  }

  // Only direct producers can be cache read
  const std::set<int>& producers = GetDirectProducers(task, state, target_stage_id);
  if (producers.find(stage_id) == producers.end()) {
    return ConditionKind::kSkip;
  }

  return ConditionKind::kApplyAndSkipRest;
}

std::vector<std::pair<State, int>> RuleAddCacheRead::Apply(const SketchPolicyNode& policy,
                                                           const State& state, int stage_id) const {
  const SearchTask& task = policy.search_task;
  const std::set<int>& consumers = GetConsumers(task, state, stage_id);
  CHECK_EQ(consumers.size(), 1);
  int target_stage_id = *consumers.begin();
  State tmp_s = state;

  // Cache read add shared memory
  int added_stage_id = tmp_s.cache_read(stage_id, "shared", {target_stage_id}, task->compute_dag);
  target_stage_id++;
  const auto& share_read_pos =
      GetLastReduceIteratorInOutermostReduceTile(tmp_s->stages[target_stage_id]);
  tmp_s.compute_at(added_stage_id, target_stage_id, share_read_pos);
  return {std::make_pair(tmp_s, stage_id)};
}

/********** RuleAddCacheWrite **********/

SketchGenerationRule::ConditionKind RuleAddCacheWrite::MeetCondition(const SketchPolicyNode& policy,
                                                                     const State& state,
                                                                     int stage_id) const {
  // Add cache write if a stage needs multi-level tiling, but does not have a element-wise
  // matched consumer
  if (NeedsMultilevelTiling(policy.search_task, state, stage_id) &&
      !HasSingleElementwiseMatchedConsumer(policy.search_task, state, stage_id)) {
    // An apply and skip rule will be handled in RuleMultiLevelTilingWithFusion
    return IsGPUTask(policy.search_task) ? ConditionKind::kApplyAndSkipRest : ConditionKind::kApply;
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
                                                                  int stage_id) const {
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
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
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

/********** RuleCrossThreadReduction **********/

SketchGenerationRule::ConditionKind RuleCrossThreadReduction::MeetCondition(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  CHECK(IsGPUTask(policy.search_task));

  // If it is an intermidiate state created by RuleAddCacheWrite,
  // we just skip it.
  if (HasCacheWriteStage(state, stage_id)) {
    return ConditionKind::kSkip;
  }

  const auto& op = state->stages[stage_id]->op;
  if (op->IsInstance<te::ComputeOpNode>()) {
    // Compute the product of lengths of all space iters and all reduce iters
    int cum_space_len, cum_reduce_len;
    std::tie(cum_space_len, cum_reduce_len) =
        GetCumulativeSpaceAndReductionLength(state->stages[stage_id]);

    if (NeedsMultilevelTiling(policy.search_task, state, stage_id)) {
      // Do rfactor if we do not have enough parallelism on space iters
      return cum_space_len < cum_reduce_len ? ConditionKind::kApply : ConditionKind::kSkip;
    } else if (cum_reduce_len > 1) {
      // Try rfactor for other reduction operators
      return cum_reduce_len > policy.search_task->hardware_params->warp_size ? ConditionKind::kApply
                                                                             : ConditionKind::kSkip;
    }
  }

  return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleCrossThreadReduction::Apply(const SketchPolicyNode& policy,
                                                                   const State& state,
                                                                   int stage_id) const {
  const SearchTask& task = policy.search_task;
  State tmp_s = state;

  // fuse all reduction iters
  Array<Iterator> space_iters, reduce_iters;
  Iterator fused_reduce_iter;
  tmp_s =
      FuseAllReductionIterators(tmp_s, stage_id, &fused_reduce_iter, &space_iters, &reduce_iters);

  // Check the opportunity for kernel fusion
  bool fusible = false;
  int target_stage_id = GetSingleConsumerId(policy.search_task, tmp_s, stage_id);
  int num_common_outer = -1;
  if (target_stage_id >= 0) {
    num_common_outer =
        GetNumCommonOuterIterator(policy.search_task, tmp_s, stage_id, target_stage_id);
    if (num_common_outer > 0 &&
        !NeedsMultilevelTiling(policy.search_task, state, target_stage_id)) {
      fusible = true;
    }
  }

  if (fusible) {
    const Stage& target_stage = state->stages[target_stage_id];
    std::vector<int> split_step_ids;

    GetSplitStepIds(tmp_s, target_stage_id, &split_step_ids);

    if (split_step_ids.size() == 0) {
      // If the target stage does not have split step,
      // it must be a simple stage without reduce iters.
      // We then should do a split for it.
      CHECK(!HasReduceIter(target_stage));
      const auto& split_res = tmp_s.split(target_stage_id, target_stage->iters.back(),
                                          {Integer(task->hardware_params->warp_size)});
      tmp_s.bind(target_stage_id, split_res[1], IteratorAnnotation::kThreadX);
      split_step_ids.push_back(tmp_s->transform_steps.size() - 2);
    }

    CHECK_EQ(split_step_ids.size(), 1);

    const Iterator& target_iter = tmp_s->stages[target_stage_id]->iters[num_common_outer - 1];
    const auto& split_res = tmp_s.follow_split(stage_id, fused_reduce_iter, split_step_ids[0], 1);
    tmp_s.bind(stage_id, split_res[1], IteratorAnnotation::kThreadX);
    tmp_s.compute_at(stage_id, target_stage_id, target_iter);
  } else {
    const auto& split_res =
        tmp_s.split(stage_id, fused_reduce_iter, {Integer(task->hardware_params->warp_size)});
    tmp_s.bind(stage_id, split_res[1], IteratorAnnotation::kThreadX);
  }

  return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** RuleSpecialComputeLocationGPU **********/

SketchGenerationRule::ConditionKind RuleSpecialComputeLocationGPU::MeetCondition(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  if (GetProducers(policy.search_task, state, stage_id).empty()) {
    return ConditionKind::kSkip;
  }

  const std::set<int>& consumers = GetConsumers(policy.search_task, state, stage_id);
  if (consumers.size() == 1 && state->stages[*consumers.begin()]->op->attrs.count(
                                   SearchPolicyKey::simplify_const_tensor_indices)) {
    return ConditionKind::kApplyAndSkipRest;
  }

  return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> RuleSpecialComputeLocationGPU::Apply(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  State tmp_s = state;
  const std::set<int>& consumers = GetConsumers(policy.search_task, state, stage_id);
  CHECK_EQ(consumers.size(), 1);

  // Get the last outer space iterator that is not unrolled.
  const Stage& target_stage = state->stages[*consumers.begin()];
  for (size_t i = 0; i < target_stage->iters.size(); ++i) {
    if (target_stage->iters[i]->annotation == IteratorAnnotation::kUnroll) {
      CHECK_GT(i, 0);

      tmp_s.compute_at(stage_id, *consumers.begin(), target_stage->iters[i - 1]);
      break;
    }
  }

  return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** Init Population **********/

PopulationGenerationRule::ResultKind InitFillTileSize::Apply(SketchPolicyNode* policy,
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

PopulationGenerationRule::ResultKind MutateComputeLocationCommon(SketchPolicyNode* policy,
                                                                 State* state,
                                                                 bool infer_bound = true) {
  if (GetIntParam(policy->params, SketchParamKey::disable_change_compute_location)) {
    return PopulationGenerationRule::ResultKind::kValid;
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

  if (infer_bound) {
    *state = policy->search_task->compute_dag.InferBound(*state);
  }
  return PopulationGenerationRule::ResultKind::kValid;
}

PopulationGenerationRule::ResultKind InitChangeComputeLocation::Apply(SketchPolicyNode* policy,
                                                                      State* state) const {
  return MutateComputeLocationCommon(policy, state, true);
}

PopulationGenerationRule::ResultKind InitParallel::Apply(SketchPolicyNode* policy,
                                                         State* state) const {
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

PopulationGenerationRule::ResultKind InitUnroll::Apply(SketchPolicyNode* policy,
                                                       State* state) const {
  std::vector<int> auto_unroll_configs = IsGPUTask(policy->search_task)
                                             ? std::vector<int>({0, 16, 64, 512, 1024})
                                             : std::vector<int>({0, 16, 64, 512});
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

PopulationGenerationRule::ResultKind InitVectorization::Apply(SketchPolicyNode* policy,
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
      // Stop if this iterator has been a compute at attach point
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

PopulationGenerationRule::ResultKind InitThreadBind::Apply(SketchPolicyNode* policy,
                                                           State* state) const {
  std::set<int> multi_level_tiling_root_set;
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    if (NeedsMultilevelTiling(policy->search_task, *state, stage_id)) {
      const Stage& stage = (*state)->stages[stage_id];
      if (stage->compute_at != ComputeAtKind::kIter) {
        // This stage is not multi-level tiled,
        // so it must be produced by RuleCrossThreadReduction.
        CHECK(HasCrossThreadReduction(*state, stage_id));
      } else {
        const auto res = (*state)->attach_map->stage_to_attach_iter.find(stage_id);
        CHECK(res != (*state)->attach_map->stage_to_attach_iter.end());
        multi_level_tiling_root_set.insert(res->second.first);
      }
    }
  }

  *state = policy->search_task->compute_dag.InferBound(*state);

  for (int stage_id = (*state)->stages.size() - 1; stage_id >= 0; --stage_id) {
    const Stage& stage = (*state)->stages[stage_id];

    if (stage->compute_at == ComputeAtKind::kInlined || stage->op_type == StageKind::kPlaceholder) {
      continue;
    }

    // Deal with the cross-thread reduction generated by RuleCrossThreadReduction
    if (HasCrossThreadReduction(*state, stage_id)) {
      Iterator fused_it;
      *state = std::move(FuseAllOuterSpaceIterators(*state, stage_id, &fused_it));
      state->bind(stage_id, fused_it, IteratorAnnotation::kBlockX);
      continue;
    }

    // Skip if this stage has already been annotaed with threadIdx.x
    if (HasAnnotatedIter(stage, IteratorAnnotation::kThreadX)) {
      continue;
    }

    if (stage->compute_at == ComputeAtKind::kRoot) {
      // This stage has not been tiled, but in GPU schedule, we must tile the root stage
      // to do thread binding
      if (!multi_level_tiling_root_set.count(stage_id)) {
        Iterator fused_it;
        *state = FuseAllOuterSpaceIterators(*state, stage_id, &fused_it);

        if (GetExtent(fused_it) <= policy->search_task->hardware_params->warp_size) {
          state->bind(stage_id, fused_it, IteratorAnnotation::kThreadX);
        } else {
          // Set threadIdx.x = default_warp_size by default.
          // The later EvolutionarySearch will try more possibility
          const auto& split_its = state->split(
              stage_id, fused_it, {Integer(policy->search_task->hardware_params->warp_size)});
          state->bind(stage_id, split_its[0], IteratorAnnotation::kBlockX);
          state->bind(stage_id, split_its[1], IteratorAnnotation::kThreadX);
        }
        continue;
      }

      // Otherwise, this is a tiled root stage, we assume it should be tiled with 3 space level
      // in the outer iterators.
      // The remaining part deals with the thread binding for multi-level tiled stages
      auto pop = stage->op.as<te::ComputeOpNode>();
      std::vector<Iterator> to_fuse;
      int total_space_extent = 1;
      for (const auto& i : pop->root_iter_vars()) {
        CHECK(i->dom.defined());
        const auto& pint = i->dom->extent.as<IntImmNode>();
        CHECK(pint);
        total_space_extent *= pint->value;
      }

      // Check if the total space extent is too small for multi-level thread binding
      if (total_space_extent <= policy->search_task->hardware_params->warp_size) {
        Iterator fused_it;
        *state = FuseAllOuterSpaceIterators(*state, stage_id, &fused_it);
        state->bind(stage_id, fused_it, IteratorAnnotation::kThreadX);
        continue;
      }

      // Fuse the outermost space tile as blockIdx
      for (size_t i = 0; i < pop->axis.size(); i++) {
        const auto& it = (*state)->stages[stage_id]->iters[i];
        // There may be some iterators that are marked with no split, stop if reaches next
        // tiling level
        if (!StrEndsWith(it->name, ".0")) {
          break;
        }
        to_fuse.push_back(it);
      }
      const auto& blockidx_it = state->fuse(stage_id, to_fuse);
      state->bind(stage_id, blockidx_it, IteratorAnnotation::kBlockX);

      // Fuse the second outermost space tile as vthread
      to_fuse.clear();
      for (size_t i = 1; i < pop->axis.size() + 1; i++) {
        const auto& it = (*state)->stages[stage_id]->iters[i];
        // There may be some iterators that are marked with no split, stop if reaches next
        // tiling level
        if (!StrEndsWith(it->name, ".1")) {
          break;
        }
        to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
      }
      const auto& vthread_it = state->fuse(stage_id, to_fuse);
      if (GetExtent(vthread_it) > policy->search_task->hardware_params->max_vthread_extent) {
        return ResultKind::kInvalid;
      }
      state->bind(stage_id, vthread_it, IteratorAnnotation::kVThread);

      // Fuse the third outermost space tile as threadIdx
      to_fuse.clear();
      for (size_t i = 2; i < pop->axis.size() + 2; i++) {
        const auto& it = (*state)->stages[stage_id]->iters[i];
        // There may be some iterators that are marked with no split, stop if reaches next
        // tiling level
        if (!StrEndsWith(it->name, ".2")) {
          break;
        }
        to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
      }
      const auto& threadidx_it = state->fuse(stage_id, to_fuse);
      if (GetExtent(threadidx_it) < policy->search_task->hardware_params->warp_size) {
        return ResultKind::kInvalid;
      }
      state->bind(stage_id, threadidx_it, IteratorAnnotation::kThreadX);
    } else if (stage->compute_at == ComputeAtKind::kIter &&
               StrEndsWith(stage->op->name, ".shared")) {
      // Do cooperative fetching for the cache read stage.
      // Get spatial_split_step_ids from the root stage
      const auto& it = (*state)->attach_map->stage_to_attach_iter.find(stage_id);
      CHECK(it != (*state)->attach_map->stage_to_attach_iter.end());
      Array<Integer> spatial_split_step_ids = GetSpatialSplitStepIds(*state, it->second.first);

      // Fuse all iterators to do cooperative fetching
      Iterator fused = state->fuse(stage_id, (*state)->stages[stage_id]->iters);
      // Split out an extra iterator for vectorization
      // The later EvolutionarySearch will try more possibility
      const auto& iters0 = state->split(stage_id, fused, {Integer(1)});
      state->vectorize(stage_id, iters0[1]);
      // Follow split to keep a same thread extent with the root stage
      const auto& iters1 =
          state->follow_fused_split(stage_id, iters0[0], spatial_split_step_ids, 1, true);
      state->bind(stage_id, iters1[1], IteratorAnnotation::kThreadX);
    }
  }
  return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind MutateTileSize::Apply(SketchPolicyNode* policy,
                                                           State* state) const {
  int max_innermost_split_factor =
      GetIntParam(policy->params, SketchParamKey::max_innermost_split_factor);

  // Extract all SplitStep
  std::vector<size_t> split_step_ids;
  for (size_t i = 0; i < (*state)->transform_steps.size(); ++i) {
    if (auto ps = (*state)->transform_steps[i].as<SplitStepNode>()) {
      if (!ps->extent.defined() || !ps->extent.value()->IsInstance<IntImmNode>()) {
        continue;
      }
      auto innermost_factor = ps->lengths.back().value_or(max_innermost_split_factor + 1);
      if (GetIntImm(innermost_factor) <= max_innermost_split_factor) {
        split_step_ids.push_back(i);
      }
    }
  }
  if (split_step_ids.empty()) {
    // No tile size could be mutated.
    return ResultKind::kInvalid;
  }

  // Select a SplitStep with extent larger than one to mutate.
  int retry_ct = 0;
  int64_t extent = 1;
  int step_id;
  const SplitStepNode* ps;

  do {
    step_id = split_step_ids[(policy->rand_gen)() % split_step_ids.size()];
    ps = (*state)->transform_steps[step_id].as<SplitStepNode>();
    CHECK(ps != nullptr);
    extent = GetIntImm(ps->extent.value());
    retry_ct += 1;
  } while (retry_ct < static_cast<int>(split_step_ids.size()) << 2 && (extent == 1 || extent == 0));

  if (extent <= 1) {
    // Cannot find a step with extent larger than one.
    return ResultKind::kInvalid;
  }

  // Fetch the current tile sizes.
  std::vector<int> lengths(ps->lengths.size() + 1, 1);
  for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
    lengths[i + 1] = GetIntImm(ps->lengths[i].value());
  }
  lengths[0] = extent / ElementProduct(lengths);

  // Random permute the tile size order.
  std::vector<int> random_perm;
  RandomPermutation(lengths.size(), &random_perm, &(policy->rand_gen));

  // Try to divide a factor from one tile size and multiple it to another.
  for (size_t i = 0; i < random_perm.size(); ++i) {
    size_t src_idx = random_perm[i];
    int length = lengths[src_idx];
    if (length <= 1) {
      continue;
    }

    size_t dst_idx = random_perm[(i + 1) % random_perm.size()];
    const std::vector<int>& factors = policy->split_memo.GetFactors(length);
    CHECK_GE(factors.size(), 1);

    int divide_factor;
    if (dst_idx == lengths.size() - 1) {
      // Maintain the restriction of hardware_params.max_innermost_split_factor.
      int max_factor_index = static_cast<int>(factors.size()) - 1;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * lengths[dst_idx] <= max_innermost_split_factor) {
          break;
        }
      }
      if (max_factor_index == 0) {
        // Failed on this dst_idx, try next one.
        continue;
      }
      divide_factor = factors[1 + (policy->rand_gen)() % (max_factor_index)];
    } else {
      divide_factor = factors[1 + (policy->rand_gen)() % (factors.size() - 1)];
    }

    // Divide one factor from lengths[src_idx] and multiply it to lengths[dst_idx].
    Array<Integer> new_lengths;
    for (size_t j = 1; j < lengths.size(); ++j) {
      if (j == src_idx) {
        new_lengths.push_back(Integer(lengths[j] / divide_factor));
      } else if (j == dst_idx) {
        new_lengths.push_back(Integer(lengths[j] * divide_factor));
      } else {
        new_lengths.push_back(Integer(lengths[j]));
      }
    }

    StateNode* pstate = state->CopyOnWrite();
    pstate->transform_steps.Set(
        step_id, SplitStep(ps->stage_id, ps->iter_id, ps->extent,
                           Array<Optional<Integer>>(new_lengths.begin(), new_lengths.end()),
                           ps->inner_to_outer));
    return ResultKind::kValid;
  }
  return ResultKind::kInvalid;
}

PopulationGenerationRule::ResultKind MutateMaxUnrollFactor::Apply(SketchPolicyNode* policy,
                                                                  State* state) const {
  // Extract all auto_unroll_max_step pragma steps.
  std::vector<int> annotate_steps;
  for (size_t i = 0; i < (*state)->transform_steps.size(); ++i) {
    if (auto ps = (*state)->transform_steps[i].as<PragmaStepNode>()) {
      if (StrStartsWith(ps->pragma_type, "auto_unroll_max_step")) {
        annotate_steps.push_back(i);
      }
    }
  }
  if (annotate_steps.empty()) {
    return ResultKind::kInvalid;
  }

  // Random pick up one unroll factor candidate.
  auto cands = (IsGPUTask(policy->search_task)) ? &gpu_unroll_cands_ : &cpu_unroll_cands_;
  auto new_factor = std::to_string((*cands)[(policy->rand_gen)() % cands->size()]);

  // Random pick up and mutate an unroll step.
  auto step_id = annotate_steps[(policy->rand_gen)() % annotate_steps.size()];
  auto ps = (*state)->transform_steps[step_id].as<PragmaStepNode>();
  CHECK(ps);
  StateNode* pstate = state->CopyOnWrite();
  pstate->transform_steps.Set(step_id,
                              PragmaStep(ps->stage_id, ps->iter_id,
                                         std::string("auto_unroll_max_step") + "$" + new_factor));
  return ResultKind::kValid;
}

PopulationGenerationRule::ResultKind MutateComputeLocation::Apply(SketchPolicyNode* policy,
                                                                  State* state) const {
  return MutateComputeLocationCommon(policy, state, false);
}

PopulationGenerationRule::ResultKind MutateParallel::Apply(SketchPolicyNode* policy,
                                                           State* state) const {
  // This mutation rule only focuses on a case that parallel was added to
  // the outermost loop and the loop is generated by fusing other loops.
  // In short, we mutate the fusion step before the parallel step.

  // Extract all parallel steps.
  std::vector<int> parallel_steps;
  for (size_t s = 0; s < (*state)->transform_steps.size(); ++s) {
    auto ps = (*state)->transform_steps[s].as<AnnotationStepNode>();
    if (!ps || ps->annotation != IteratorAnnotation::kParallel) {
      continue;
    }

    // Skip non-outermost loop or the parallel step without fusion beforehand.
    if (ps->iter_id > 0 || s == 0 || !(*state)->transform_steps[s - 1].as<FuseStepNode>()) {
      continue;
    }
    parallel_steps.push_back(s);
  }
  if (parallel_steps.empty()) {
    return ResultKind::kInvalid;
  }

  // Randomly pick one parallel step.
  size_t step_id = parallel_steps[(policy->rand_gen)() % parallel_steps.size()];
  auto ps = (*state)->transform_steps[step_id].as<AnnotationStepNode>();
  CHECK(ps);
  size_t stage_id = ps->stage_id;
  size_t iter_id = ps->iter_id;
  const Stage& stage = (*state)->stages[stage_id];
  const Iterator& it = stage->iters[iter_id];

  // Replay a new state until the picked fuse step.
  State tmp_s = policy->search_task->compute_dag->init_state;
  for (size_t s = 0; s < step_id - 1; ++s) {
    auto step = (*state)->transform_steps[s];
    tmp_s.CopyOnWrite()->transform_steps.push_back(step);
    StepApplyToState(step, &tmp_s, policy->search_task->compute_dag);
  }

  // Determine the fusion mutation direction.
  // 0: fuse less; 1: fuse more.
  auto fuse_step = (*state)->transform_steps[step_id - 1].as<FuseStepNode>();
  auto fused_ids = fuse_step->fused_ids;
  std::vector<double> fuse_dir = {0.5, 1.0};

  // The case that we can only fuse more. This may happen after multiple mutations.
  if (fused_ids.size() == 1) {
    fuse_dir[0] = 0.0;
  }

  // The cases that we cannot fuse the next iters.
  if ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id)) ||
      it->iter_kind == IteratorKind::kReduction || it->annotation != IteratorAnnotation::kNone) {
    if (fuse_dir[0] == 0.0) {
      // No room to mutate this fusion.
      return ResultKind::kInvalid;
    }
    fuse_dir[0] = 1.0;
  }

  // Mutate the fusion iters and replay the mutated fused/annotation steps.
  int iter_offset = 0;
  if (RandomChoose(fuse_dir, &(policy->rand_gen)) == 0) {
    fused_ids.pop_back();
    iter_offset = 1;
  } else {
    auto last_id = fused_ids.back().get()->value;
    fused_ids.push_back(last_id + 1);
    iter_offset = -1;
  }
  auto new_fuse_step = FuseStep(stage_id, fused_ids);
  tmp_s.CopyOnWrite()->transform_steps.push_back(new_fuse_step);
  StepApplyToState(new_fuse_step, &tmp_s, policy->search_task->compute_dag);
  tmp_s.CopyOnWrite()->transform_steps.push_back((*state)->transform_steps[step_id]);
  StepApplyToState((*state)->transform_steps[step_id], &tmp_s, policy->search_task->compute_dag);

  // Replay the rest steps.
  for (size_t s = step_id + 1; s < (*state)->transform_steps.size(); ++s) {
    auto step = (*state)->transform_steps[s];
    if (step->stage_id == static_cast<int>(stage_id)) {
      // Since we changed the loop structure, iter ID in later steps to the same stage
      // has to be adjusted.
      auto ps = step.as<AnnotationStepNode>();
      if (ps) {
        if (ps->iter_id == 0) {
          step = AnnotationStep(ps->stage_id, 0, ps->annotation);
        } else {
          CHECK_LE(ps->iter_id + iter_offset, tmp_s->stages[stage_id]->iters.size());
          step = AnnotationStep(ps->stage_id, ps->iter_id + iter_offset, ps->annotation);
        }
      } else {
        // Unexpected step node that we did not process for now.
        return ResultKind::kInvalid;
      }
    }
    tmp_s.CopyOnWrite()->transform_steps.push_back(step);
    StepApplyToState(step, &tmp_s, policy->search_task->compute_dag);
  }

  *state = tmp_s;
  return ResultKind::kValid;
}

}  // namespace auto_scheduler
}  // namespace tvm
