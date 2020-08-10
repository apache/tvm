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
 * \file auto_scheduler/search_policy/sketch_search_policy.h
 * \brief The search policy that searches in a hierarchical search space defined by sketches.
 * The policy randomly samples programs from the space defined by sketches
 * and use evolutionary search to fine-tune them.
 */

#include "sketch_search_policy.h"

#include <tvm/runtime/registry.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(SketchSearchPolicyNode);

/********** Sketch Generation Rule **********/

// The rule that simply skips the current stage(return a unchanged state and try the next stage).
class RuleSkipStage : public SketchGenerationRule {
 public:
  ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                              int stage_id) final {
    // This rule should be the last rule, always return true to decrease the stage index count
    return ConditionKind::kApply;
  }

  std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy, const State& state,
                                           int stage_id) const final {
    return {std::make_pair(state, stage_id - 1)};
  }
};

// The rule that inlines simple elementwise ops.
// Notice: This rule only inlines the strictly inlineable stages. Stages marked as not strictly
// inlineable will have a chance to try different compute at location in InitPopulation later.
class RuleAlwaysInline : public SketchGenerationRule {
 public:
  ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                              int stage_id) final {
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

  std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy, const State& state,
                                           int stage_id) const final {
    State tmp_s = state;
    tmp_s.compute_inline(stage_id);
    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
  }
};

// The rule that performs multi-level tiling.
class RuleMultiLevelTiling : public SketchGenerationRule {
 public:
  ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                              int stage_id) final {
    return NeedsMultilevelTiling(policy.search_task, state, stage_id)
               ? ConditionKind::kApplyAndSkipRest
               : ConditionKind::kSkip;
  }

  std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy, const State& state,
                                           int stage_id) const final {
    // TODO(jcf94): Add support for GPU structure when introducing GPU search policy.
    const std::string& multi_level_tiling_structure =
        GetStringParam(policy.params, SketchParamKey::MultiLevelTiling::cpu_structure);
    State tmp_s = DoMultiLevelTiling(state, stage_id, multi_level_tiling_structure);
    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
  }
};

// The rule that performs multi-level tiling and fuses later consumers.
class RuleMultiLevelTilingWithFusion : public SketchGenerationRule {
 public:
  ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                              int stage_id) final {
    if (NeedsMultilevelTiling(policy.search_task, state, stage_id) &&
        HasSingleElementwiseMatchedConsumer(policy.search_task, state, stage_id,
                                            &target_stage_id)) {
      // Always do fusion for stage with cache_write
      // TODO(jcf94): Always do fusion on GPU when introducing GPU search policy.
      return HasCacheWriteStage(state, stage_id) ? ConditionKind::kApplyAndSkipRest
                                                 : ConditionKind::kApply;
    }
    return ConditionKind::kSkip;
  }

  std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy, const State& state,
                                           int stage_id) const final {
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

 private:
  int target_stage_id;
};

// The rule that adds a cache write stage.
class RuleAddCacheWrite : public SketchGenerationRule {
 public:
  ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                              int stage_id) final {
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

  std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy, const State& state,
                                           int stage_id) const final {
    State tmp_s = state;
    tmp_s.cache_write(stage_id, "local", policy.search_task->compute_dag);
    return {std::make_pair(std::move(tmp_s), stage_id)};
  }
};

// The rule that adds rfactor stage.
class RuleAddRfactor : public SketchGenerationRule {
 public:
  ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                              int stage_id) final {
    return (NeedsRfactor(policy.search_task, state, stage_id) &&
            !HasCacheWriteStage(state, stage_id))
               ? ConditionKind::kApply
               : ConditionKind::kSkip;
  }

  std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy, const State& state,
                                           int stage_id) const final {
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
};

// The rule that deals with compute ops that perform "fake reduction" with const tensors.
// This kind of op comes from winograd transformation.
class RuleSimplifyComputeWithConstTensor : public SketchGenerationRule {
 public:
  ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                              int stage_id) final {
    return state->stages[stage_id]->op->attrs.count(SearchPolicyKey::simplify_const_tensor_indices)
               ? ConditionKind::kApplyAndSkipRest
               : ConditionKind::kSkip;
  }

  std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy, const State& state,
                                           int stage_id) const final {
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
};

static RuleSkipStage rule_skip_stage;
static RuleAlwaysInline rule_always_inline;
static RuleMultiLevelTiling rule_multi_level_tiling;
static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
static RuleAddCacheWrite rule_add_cache_write_stage;
static RuleAddRfactor rule_add_rfactor;
static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;

/********** Init Population **********/

// The rule that fills the incomplete SplitSteps.
class InitFillTileSize : public InitPopulationRule {
 public:
  ResultKind Apply(SketchSearchPolicyNode* policy, State* state) const final {
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
        const auto& candidate_lengths =
            candidate_lens[(policy->rand_gen)() % candidate_lens.size()];

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
};

// The rule that randomly changes the computation location for some stages, which do not need
// tiling and are not strictly inlineable(e.g. data padding).
class InitChangeComputeLocation : public InitPopulationRule {
 public:
  ResultKind Apply(SketchSearchPolicyNode* policy, State* state) const {
    if (GetIntParam(policy->params, SketchParamKey::disable_change_compute_location)) {
      return ResultKind::kValid;
    }

    for (int stage_id = static_cast<int>((*state)->stages.size()) - 1; stage_id >= 0; stage_id--) {
      const Stage& stage = (*state)->stages[stage_id];
      // Skip the inlined stages and placeholders
      if (stage->op_type == StageKind::kPlaceholder ||
          stage->compute_at == ComputeAtKind::kInlined) {
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

        if ((*state)->attach_map->iter_to_attached_stages.count(
                std::make_pair(target_stage_id, i))) {
          break;
        }
      }

      // if the target_stage is already compute_at another stage X, try also compute_at X
      // We call stage X as `target_target_stage`
      if (target_compute_at_other) {
        int target_target_stage_id;
        target_target_stage_id =
            (*state)->attach_map->stage_to_attach_iter.at(target_stage_id).first;
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
};

// The rule that annotates parallel for CPU.
class InitParallel : public InitPopulationRule {
 public:
  ResultKind Apply(SketchSearchPolicyNode* policy, State* state) const {
    std::function<void(const SketchSearchPolicyNode&, State*, int stage_id, int iter_offset)>
        annotate_parallel;
    annotate_parallel = [&annotate_parallel](const SketchSearchPolicyNode& policy, State* state,
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

        if ((*state)->attach_map->iter_to_attached_stages.count(
                std::make_pair(stage_id, iter_id))) {
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
};

class InitUnroll : public InitPopulationRule {
 public:
  ResultKind Apply(SketchSearchPolicyNode* policy, State* state) const {
    std::vector<int> auto_unroll_configs = {0, 16, 64, 512};
    for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
      const Stage& stage = (*state)->stages[stage_id];
      // Skip the inlined stage and placeholder stage
      if (stage->compute_at == ComputeAtKind::kInlined ||
          stage->op_type == StageKind::kPlaceholder) {
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
};

class InitVectorization : public InitPopulationRule {
 public:
  ResultKind Apply(SketchSearchPolicyNode* policy, State* state) const {
    for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
      const Stage& stage = (*state)->stages[stage_id];
      // Skip the inlined stage and placeholder stage
      if (stage->compute_at == ComputeAtKind::kInlined ||
          stage->op_type == StageKind::kPlaceholder) {
        continue;
      }

      // Try to fuse and vectorize the space iterators in the inner most tile
      int64_t cum_length_prod = 1;

      int num_fusible = 0;
      while (num_fusible < static_cast<int>(stage->iters.size())) {
        int iter_id = static_cast<int>(stage->iters.size()) - 1 - num_fusible;
        // Stop if this iterator has been a compute at attatch point
        if ((*state)->attach_map->iter_to_attached_stages.count(
                std::make_pair(stage_id, iter_id))) {
          break;
        }

        const Iterator& it = stage->iters[iter_id];
        // Stop if we meet a reduce iterator or annotated iterator
        if (it->iter_kind == IteratorKind::kReduction ||
            it->annotation != IteratorAnnotation::kNone) {
          break;
        }

        // Stop if the memory access is not continuous (vectorizable)
        // This check is not so easy to do, so we ....... [here]
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
};

static InitFillTileSize init_fill_tile_size;
static InitChangeComputeLocation init_change_compute_location;
static InitParallel init_parallel;
static InitUnroll init_unroll;
static InitVectorization init_vectorization;

/********** Sketch Search Policy **********/

SketchSearchPolicy::SketchSearchPolicy(SearchTask task, CostModel schedule_cost_model,
                                       Map<String, ObjectRef> params, int seed, int verbose,
                                       Optional<Array<SearchCallback>> init_search_callbacks) {
  auto node = make_object<SketchSearchPolicyNode>();
  node->search_task = std::move(task);
  node->schedule_cost_model = std::move(schedule_cost_model);
  node->rand_gen = std::mt19937(seed);
  node->params = std::move(params);
  node->verbose = verbose;

  if (init_search_callbacks) {
    PrintTitle("Call init-search callbacks", verbose);
    // Candidates:
    // - auto_scheduler.PreloadMeasuredStates: Load already measured states to
    //   `measured_states_set_`, `measured_states_vector_` and `measured_states_throughputs_`.
    // - auto_scheduler.PreloadCustomSketchRule: Add user custom sketch rules to `sketch_rules`,
    //   these rules will be processed prior to the default rules.
    node->RunCallbacks(init_search_callbacks.value());
  }

  // The default sketch rules for CPU policy
  // Notice: Some rules require us to skip all the rest rules after they are applied.
  // So the rules below should be ordered carefully.
  node->sketch_rules.push_back(&rule_always_inline);
  node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
  node->sketch_rules.push_back(&rule_add_rfactor);
  node->sketch_rules.push_back(&rule_add_cache_write_stage);
  node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
  node->sketch_rules.push_back(&rule_multi_level_tiling);
  node->sketch_rules.push_back(&rule_skip_stage);  // This should always be the last rule

  // The default init population rules for CPU policy
  node->init_rules.push_back(&init_fill_tile_size);
  node->init_rules.push_back(&init_change_compute_location);
  node->init_rules.push_back(&init_parallel);
  node->init_rules.push_back(&init_unroll);
  node->init_rules.push_back(&init_vectorization);

  data_ = std::move(node);
}

State SketchSearchPolicyNode::Search(int n_trials, int early_stopping, int num_measure_per_iter,
                                     ProgramMeasurer measurer) {
  num_measure_per_iter_ = num_measure_per_iter;

  if (n_trials <= 1) {
    // No measurement is allowed
    const Array<State>& best_states = SearchOneRound(0);
    CHECK_GT(best_states.size(), 0);
    return best_states[0];
  } else {
    int num_random =
        static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy) * num_measure_per_iter);
    early_stopping = early_stopping < 0 ? std::numeric_limits<int>::max() >> 1 : early_stopping;
    measurer->Reset();

    int ct = 0;
    Array<MeasureInput> inputs;
    Array<MeasureResult> results;
    while (ct < n_trials) {
      if (!inputs.empty()) {
        // Retrain cost models before the next search round
        PrintTitle("Train cost model", verbose);
        schedule_cost_model->Update(inputs, results);
      }

      // Search one round to get promising states
      PrintTitle("Search", verbose);
      Array<State> random_states;
      Array<State> best_states = SearchOneRound(num_random, &random_states);

      // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
      best_states = search_task->compute_dag.InferBound(best_states);
      random_states = search_task->compute_dag.InferBound(random_states);

      // Pick `num_measure_per_iter` states to measure, check hash to remove already measured state
      // Also pick some random states to do eps-greedy
      inputs = PickStatesWithEpsGreedy(best_states, random_states, n_trials - ct);

      // Have traversed all of the search space
      if (inputs.empty()) {
        StdCout(verbose) << "All candidates in the search space have been measured." << std::endl;
        break;
      }

      // Measure candidate states
      PrintTitle("Measure", verbose);
      measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs, &results);
      ct += inputs.size();

      // Check if reach the early stopping condition
      if (ct - measurer->best_ct[search_task->workload_key] > early_stopping) {
        StdCout(verbose) << "Stop early since no performance improvement in the last "
                         << early_stopping << " measure steps.\n";
        break;
      }

      // Update measured states throughputs. These states will join the EvolutionarySearch in later
      // search rounds.
      for (const auto& res : results) {
        measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
      }
    }
    PrintTitle("Done", verbose);

    return measurer->best_state[search_task->workload_key];
  }
}

Array<State> SketchSearchPolicyNode::SearchOneRound(int num_random_states,
                                                    Array<State>* random_states) {
  // Temporal object to be used if the input pointer is nullptr
  Array<State> temp_random_states;
  if (random_states == nullptr) {
    random_states = &temp_random_states;
  } else {
    random_states->clear();
  }

  // Get parameters
  int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
  int num_use_measured =
      std::min(static_cast<int>(measured_states_vector_.size()),
               static_cast<int>(
                   GetDoubleParam(params, SketchParamKey::EvolutionarySearch::use_measured_ratio) *
                   population));
  bool is_cost_model_reasonable = !schedule_cost_model->IsInstance<RandomModelNode>();

  // 1. Generate sketches
  const Array<State>& sketches = GenerateSketches();

  // 2. Sample the init population
  Array<State> init_populations = SampleInitPopulation(
      sketches, is_cost_model_reasonable ? population - num_use_measured : population);

  // 3. If the cost model is useless (i.e. RandomCostModel), just random pick some generated
  // states, else perform evolutionary search
  if (is_cost_model_reasonable) {
    // Also insert already measured good states to the initial population
    std::vector<int> indices = Argsort(measured_states_throughputs_);
    for (int i = 0; i < num_use_measured; i++) {
      init_populations.push_back(measured_states_vector_[indices[i]]);
    }
    // Sample some random states for eps-greedy
    *random_states = RandomSampleStates(init_populations, &rand_gen, num_random_states * 10);
    return EvolutionarySearch(init_populations, num_measure_per_iter_ * 2);
  } else {
    return RandomSampleStates(init_populations, &rand_gen, num_measure_per_iter_ * 3);
  }
}

Array<State> SketchSearchPolicyNode::GenerateSketches() {
  State init_state = search_task->compute_dag->init_state;

  // Two ping pong buffers to avoid copy
  Array<State> states_buf1{init_state}, states_buf2;
  Array<State>* pnow = &states_buf1;
  Array<State>* pnext = &states_buf2;

  // A map that maps state to its current working position (stage_id)
  std::unordered_map<State, int, ObjectHash, ObjectEqual> cur_stage_id_map;
  cur_stage_id_map[init_state] = static_cast<int>(init_state->stages.size() - 1);

  // Derivation rule based enumeration
  Array<State> out_states;
  while (!pnow->empty()) {
    pnext->clear();
    for (const State& state : *pnow) {
      int stage_id = cur_stage_id_map[state];

      // Reaches to the terminal stage
      if (stage_id < 0) {
        out_states.push_back(state);
        continue;
      }

      // Try all derivation rules
      for (const auto& rule : sketch_rules) {
        auto cond = rule->MeetCondition(*this, state, stage_id);
        if (cond != SketchGenerationRule::ConditionKind::kSkip) {
          for (const auto& pair : rule->Apply(*this, state, stage_id)) {
            cur_stage_id_map[pair.first] = pair.second;
            pnext->push_back(pair.first);
          }
          // Skip the reset rules
          if (cond == SketchGenerationRule::ConditionKind::kApplyAndSkipRest) {
            break;
          }
        }
      }
    }
    std::swap(pnow, pnext);
  }

  // Hack for rfactor: Replace the split factor for rfactor to the undefined Expr(),
  // so later we can sample random value for the split factor.
  // Why don't we use Expr() when doing the split for rfactor at the first time?
  // Because during ApplySteps, a rfactor with undefined Expr() will crash TVM.
  // So rfactor with undefined Expr() will conflict with cache_write, cache_read, rfactor
  // in other stages
  for (size_t i = 0; i < out_states.size(); ++i) {
    auto state = out_states[i];
    auto pstate = state.CopyOnWrite();
    for (size_t step_id = 0; step_id < pstate->transform_steps.size(); ++step_id) {
      if (pstate->transform_steps[step_id]->IsInstance<RfactorStepNode>()) {
        CHECK_GE(step_id, 1);
        int split_step_id = static_cast<int>(step_id - 1);
        auto step = pstate->transform_steps[split_step_id].as<SplitStepNode>();
        CHECK(step != nullptr);
        pstate->transform_steps.Set(
            split_step_id, SplitStep(step->stage_id, step->iter_id, step->extent, {NullOpt},
                                     step->inner_to_outer));
      }
    }
    out_states.Set(i, std::move(state));
  }

  StdCout(verbose) << "Generate Sketches\t\t#s: " << out_states.size() << std::endl;
  return out_states;
}

Array<State> SketchSearchPolicyNode::SampleInitPopulation(const Array<State>& sketches,
                                                          int out_size) {
  int fail_ct = 0;
  Array<State> out_states;
  auto tic_begin = std::chrono::high_resolution_clock::now();

  // TODO(jcf94, merrymercy): Use parallel_for to run this loop in parallel
  while (static_cast<int>(out_states.size()) < out_size && fail_ct < static_cast<int>(out_size)) {
    // Random choose a starting sketch
    // TODO(jcf94, merrymercy): Maybe choose sketches in different possibility for they may have
    // different potential on generating state with better performance
    State tmp_s = sketches[(rand_gen)() % sketches.size()];

    // Derivation rule based enumeration
    bool valid = true;
    for (const auto& rule : init_rules) {
      if (rule->Apply(this, &tmp_s) == InitPopulationRule::ResultKind::kInvalid) {
        valid = false;
        break;
      }
    }

    if (valid) {
      out_states.push_back(std::move(tmp_s));
    } else {
      fail_ct++;
    }
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin)
                        .count();
  StdCout(verbose) << "Sample Initial Population\t#s: " << out_states.size()
                   << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                   << std::setprecision(2) << duration << std::endl;
  return out_states;
}

Array<State> SketchSearchPolicyNode::EvolutionarySearch(const Array<State>& init_populations,
                                                        int out_size) {
  Array<State> best_states;
  auto tic_begin = std::chrono::high_resolution_clock::now();

  // TODO(comaniac, merrymercy, jcf94): Since we haven't finished porting the cost model part
  // yet, currently delete the implementation of EvolutionarySearch. To be added later.

  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin)
                        .count();
  StdCout(verbose) << "EvolutionarySearch\t\t#s: " << best_states.size()
                   << "\tTime elapsed: " << std::fixed << std::setprecision(2) << duration
                   << std::endl;
  return best_states;
}

Array<MeasureInput> SketchSearchPolicyNode::PickStatesWithEpsGreedy(
    const Array<State>& best_states, const Array<State>& random_states, int remaining_n_trials) {
  int num_random =
      static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy) * num_measure_per_iter_);
  int num_good = num_measure_per_iter_ - num_random;

  Array<MeasureInput> inputs;
  size_t offset_best = 0, offset_random = 0;

  while (static_cast<int>(inputs.size()) < std::min(num_measure_per_iter_, remaining_n_trials)) {
    State state;

    bool has_best = offset_best < best_states.size();
    bool has_random = offset_random < random_states.size();

    if (static_cast<int>(inputs.size()) < num_good) {
      // prefer best states
      if (has_best) {
        state = best_states[offset_best++];
      } else if (has_random) {
        state = random_states[offset_random++];
      } else {
        break;
      }
    } else {
      // prefer random states
      if (has_random) {
        state = random_states[offset_random++];
      } else if (has_best) {
        state = best_states[offset_best++];
      } else {
        break;
      }
    }

    // Check if it has already been measured
    std::string state_str = state.ToStr();
    if (!measured_states_set_.count(state_str)) {
      measured_states_set_.insert(std::move(state_str));
      measured_states_vector_.push_back(state);
      inputs.push_back(MeasureInput(search_task, state));
    }
  }

  return inputs;
}

TVM_REGISTER_GLOBAL("auto_scheduler.SketchSearchPolicy")
    .set_body_typed([](SearchTask task, CostModel schedule_cost_model,
                       Map<String, ObjectRef> params, int seed, int verbose,
                       Optional<Array<SearchCallback>> init_search_callbacks) {
      return SketchSearchPolicy(task, schedule_cost_model, params, seed, verbose,
                                init_search_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchSearchPolicyGenerateSketches")
    .set_body_typed([](SketchSearchPolicy policy) { return policy->GenerateSketches(); });

}  // namespace auto_scheduler
}  // namespace tvm
