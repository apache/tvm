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
 * \file ansor/search_policy/sketch_search_policy.h
 * \brief The search policy that searches in a hierarchical search space defined by sketches.
 * The policy randomly samples programs from the space defined by sketches
 * and use evolutionary search to fine-tune them.
 */

#include "sketch_search_policy.h"
#include <tvm/runtime/registry.h>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "utils.h"

#define IS_GPU(task) ((task)->target->device_type == kDLGPU || \
  (task)->target->device_type == kDLOpenCL)

namespace tvm {
namespace ansor {

TVM_REGISTER_NODE_TYPE(SketchSearchPolicyNode);
TVM_REGISTER_OBJECT_TYPE(PreloadCustomSketchRuleNode);

// All possible candidates for auto_unroll
const std::vector<int> SketchSearchPolicyNode::auto_unroll_configs{0, 16, 64, 512, 1024};

SketchSearchPolicy::SketchSearchPolicy(CostModel program_cost_model,
                                       Map<String, ObjectRef> params,
                                       int seed) {
  auto node = make_object<SketchSearchPolicyNode>();
  node->program_cost_model = std::move(program_cost_model);
  node->rand_gen_ = std::mt19937(seed);
  node->params = std::move(params);
  data_ = std::move(node);
}

State SketchSearchPolicyNode::Search(SearchTask task, int n_trials,
    int early_stopping, int num_measure_per_iter, int verbose,
    ProgramMeasurer measurer, Array<SearchCallback> pre_search_callbacks) {
  std::vector<State> best_states, random_states;
  this->cur_task = task;
  this->verbose = verbose;
  num_measure_per_iter_ = num_measure_per_iter;

  RunCallbacks(pre_search_callbacks);

  if (n_trials <= 1) {  // no measurement is allowed
    SearchOneRound(&best_states, 0, &random_states);
    CHECK_GT(best_states.size(), 0);
    return best_states[0];
  } else {
    std::vector<MeasureInput> inputs;
    std::vector<MeasureResult> results;
    int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure_per_iter);

    measurer->Reset();

    early_stopping = early_stopping < 0 ? std::numeric_limits<int>::max() >> 1 : early_stopping;

    int ct = 0;
    while (ct < n_trials) {
      if (!inputs.empty()) {
        // retrain cost models
        PrintTitle("Train cost model", verbose);
        program_cost_model->Update(inputs, results);
      }

      // Search one round to get promising states
      PrintTitle("Search", verbose);
      SearchOneRound(&best_states, num_random, &random_states);

      // Fill correct bound.This is necessary for computing the correct ToStr() for reduncency check
      cur_task->compute_dag.InferBound(&best_states);
      cur_task->compute_dag.InferBound(&random_states);

      // Pick `num_measure_per_iter` states to measure, check hash to remove already measured state
      // Also pick some random states to do eps-greedy
      PickStatesWithEpsGreedy(&inputs, best_states, random_states, n_trials - ct);

      // Have traversed all of search space
      if (inputs.empty()) {
        StdCout(verbose) << "All candidates in the search space have been measured." << std::endl;
        break;
      }

      // Measure candidate states
      PrintTitle("Measure", verbose);
      measurer->Measure(cur_task, GetRef<SearchPolicy>(this), inputs, &results);
      ct += inputs.size();

      if (ct - measurer->best_ct[cur_task->workload_key] > early_stopping) {
        StdCout(verbose) << "Meet the early stopping condition." << std::endl;
        break;
      }

      // Update measured states. These states will join the LocalMutation in later rounds
      for (const auto& res : results) {
        measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
      }
    }
    PrintTitle("Done", verbose);

    return measurer->best_state[cur_task->workload_key];
  }
}

std::pair<Array<MeasureInput>, Array<MeasureResult> >
    SketchSearchPolicyNode::ContinueSearchOneRound(
    SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) {
  if (cur_task.defined()) {
    CHECK_EQ(cur_task, task);
  } else {
    cur_task = task;
  }
  this->verbose = verbose;
  num_measure_per_iter_ = num_measure;

  std::vector<State> best_states, random_states;
  std::vector<MeasureInput> inputs;
  std::vector<MeasureResult> results;
  int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure);

  // Search one round to get promising states
  PrintTitle("Search", verbose);
  SearchOneRound(&best_states, num_random * 2, &random_states);

  // Fill correct bound. This is necessary for computing the correct ToStr() for reduncency check
  cur_task->compute_dag.InferBound(&best_states);
  cur_task->compute_dag.InferBound(&random_states);

  // Pick `num_measure` states to measure, check hash to remove already measured state
  // Also pick some random states to do eps-greedy
  PickStatesWithEpsGreedy(&inputs, best_states, random_states, num_measure);

  // Measure candidate states
  PrintTitle("Measure", verbose);
  measurer->Measure(cur_task, GetRef<SearchPolicy>(this), inputs, &results);

  // Update throughputs of measured states. These states will join the LocalMutation in later rounds
  for (const auto& res : results) {
    measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
  }

  // Update the cost model
  Array<MeasureInput> inputs_arr(std::make_move_iterator(inputs.begin()),
                                 std::make_move_iterator(inputs.end()));
  Array<MeasureResult> results_arr(std::make_move_iterator(results.begin()),
                                   std::make_move_iterator(results.end()));

  PrintTitle("Train cost model", verbose);
  program_cost_model->Update(inputs_arr, results_arr);
  return std::make_pair(std::move(inputs_arr), std::move(results_arr));
}

void SketchSearchPolicyNode::PickStatesWithEpsGreedy(
    std::vector<MeasureInput>* inputs,
    const std::vector<State>& best_states,
    const std::vector<State>& random_states,
    int remaining_n_trials) {
  int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure_per_iter_);
  int num_good = num_measure_per_iter_ - num_random;

  inputs->clear();
  size_t offset_best = 0, offset_random = 0;

  while (static_cast<int>(inputs->size()) < std::min(num_measure_per_iter_, remaining_n_trials)) {
    const State* pstate;

    bool has_best = offset_best < best_states.size();
    bool has_random = offset_random < random_states.size();

    if (static_cast<int>(inputs->size()) < num_good) {
      // prefer best states
      if (has_best) {
        pstate = &best_states[offset_best++];
      } else if (has_random) {
        pstate = &random_states[offset_random++];
      } else {
        break;
      }
    } else {
      // prefer random states
      if (has_random) {
        pstate = &random_states[offset_random++];
      } else if (has_best) {
        pstate = &best_states[offset_best++];
      } else {
        break;
      }
    }

    // Check if it has already been measured
    std::string state_str = pstate->ToStr();

    if (measured_states_set_.count(state_str)) { continue; }
    measured_states_set_.insert(state_str);

    inputs->push_back(MeasureInput(cur_task, *pstate));
    measured_states_vector_.push_back(std::move(*pstate));
  }
}

void SketchSearchPolicyNode::SearchOneRound(std::vector<State>* best_states,
    int num_random_states, std::vector<State>* random_states) {
  best_states->clear();
  random_states->clear();

  // Get parameters
  int population = GetIntParam(params, "evolutionary_search_population");
  int num_use_measured = std::min(static_cast<int>(measured_states_vector_.size()),
      static_cast<int>(
          GetDoubleParam(params, "evolutionary_search_use_measured_ratio") * population));
  bool have_cost_model = !program_cost_model->IsInstance<RandomModelNode>();

  if (!have_cost_model) {
    num_use_measured = 0;
  }

  // Generate sketches
  std::vector<State> sketches;
  GenerateSketch(&sketches);

  // PrintAllStates(sketches);
  // exit(0);

  // Sample the init population
  std::vector<State> init_population;
  SampleInitPopulation(sketches, population - num_use_measured, &init_population);

  // PrintAllStates(init_population);
  // exit(0);

  if (have_cost_model) {
    // Also insert already measured good states to the initial population
    std::vector<int> indices;
    Argsort(measured_states_throughputs_, &indices);
    for (int i = 0; i < num_use_measured; i++) {
      init_population.push_back(measured_states_vector_[indices[i]]);
    }

    // Perform evolutionary search
    EvolutionarySearch(init_population, num_measure_per_iter_ * 2, best_states);
  } else {
    // If the cost model is useless (i.e. RandomCostModel), skip evolutionary search
    RandomSampleStates(init_population, &rand_gen_, num_measure_per_iter_ * 3, best_states);
  }

  // Sample some random states for eps-greedy
  RandomSampleStates(init_population, &rand_gen_, num_random_states * 10, random_states);
}

// The baseclass of derivation rules used in sketch generation
class SketchGenerationRule {
 public:
  enum ConditionEnum {
    kPass, kApply, kApplyAndSkipRest
  };

  virtual ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) = 0;
  virtual std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) = 0;
};

static inline bool ShouldBeCacheRead(
    const SketchSearchPolicyNode* policy, const State& state, int stage_id) {
  const SearchTask& task = policy->cur_task;
  const Stage& stage = state->stages[stage_id];

  if (HasAttrsFlag(state, stage_id,
                   SearchPolicyNode::no_cache_read_key)) {
    return false;
  }

  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;
  GetConsumers(task, state, stage->op, &consumers);
  if (consumers.size() != 1) {
    return false;
  }

  int target_stage_id = OperationToStage(*consumers.begin(), state);
  if (!NeedsMultilevelTiling(task, state,
                             state->stages[target_stage_id]->op)) {
    return false;
  }

  std::unordered_set<te::Operation, ObjectHash, ObjectEqual> producers;
  GetProducers(task, state, state->stages[target_stage_id]->op, &producers);
  // Only those directly mapped stages can do CacheRead
  if (producers.find(stage->op) == producers.end()) {
    return false;
  }

  return true;
}

static inline bool ShouldAlwaysBeInlined(
    const SketchSearchPolicyNode* policy, const State& state, int stage_id) {
  const SearchTask& task = policy->cur_task;
  const Stage& stage = state->stages[stage_id];

  if (stage->op->IsInstance<te::PlaceholderOpNode>()) {
    return false;
  }

  // Inline limitation of TVM
  if (!IsOutputOp(task, state, stage->op) && !HasReduceIter(stage)) {
    // Always inline condition:
    // 1. Has attrs that this must be inlined
    // 2. Analyse shows this is strict inlineable
    // 3. A GPU stage can be inlined(If it should be cache read, do it first)
    if (HasAttrsFlag(state, stage_id,
                     SearchPolicyNode::always_compute_inline_key) ||
        IsStrictInlineable(task, state, stage->op) ||
        (IS_GPU(policy->cur_task) &&
         !ShouldBeCacheRead(policy, state, stage_id))) {
      return true;
    }
  }

  return false;
}

// The rule that inlines simple elementwise ops
class RuleAlwaysInline : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    return ShouldAlwaysBeInlined(policy, state, stage_id) ?
        kApplyAndSkipRest : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    State tmp_s = state;
    tmp_s.compute_inline(stage_id);
    return {std::make_pair(std::move(tmp_s), stage_id - 1)};
  }
};

// The rule that simply skip the current stage
class RuleSkipStage : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];

    const auto& attrs = stage->op->attrs;
    if ((attrs.count(SearchPolicyNode::no_split_at_inner_key) ||
        attrs.count(SearchPolicyNode::no_split_at_outer_key)) &&
        NeedsMultilevelTiling(task, state, stage->op)) {
      // for the transform stages in Winograd
      return kPass;
    }

    return kApply;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    return {std::make_pair(state, stage_id - 1)};
  }
};

// The rule that performs multi-level tiling
class RuleMultiLevelTiling : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];

    return NeedsMultilevelTiling(task, state, stage->op) ?
           (IS_GPU(policy->cur_task) ? kApplyAndSkipRest : kApply) : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    std::string multi_level_tiling_structure = IS_GPU(policy->cur_task) ?
        GetStringParam(policy->params, "gpu_multi_level_tiling_structure") :
        GetStringParam(policy->params, "cpu_multi_level_tiling_structure");

    std::vector<int> spatial_split_step_ids;
    State tmp_s = state;
    tmp_s = DoMultiLevelTiling(tmp_s, stage_id, multi_level_tiling_structure,
        &spatial_split_step_ids);
    return {std::make_pair(std::move(tmp_s), stage_id-1)};
  }
};

// The rule that performs multi-level tiling and fuses later consumers
class RuleMultiLevelTilingWithFusion : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];

    int target_stage_id;

    if (IS_GPU(policy->cur_task)) {
      return NeedsMultilevelTiling(task, state, stage->op) &&
             HasSingleElementwiseMatchedConsumer(task, state, stage,
                                                 &target_stage_id) &&
             (!HasCacheReadStage(state, stage_id) ||
              HasCacheWriteStage(state, stage_id)) ?
          kApplyAndSkipRest : kPass;
    }

    return NeedsMultilevelTiling(task, state, stage->op) &&
            HasSingleElementwiseMatchedConsumer(task, state, stage,
                                                &target_stage_id) ?
        kApply : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];
    std::string multi_level_tiling_structure = IS_GPU(policy->cur_task) ?
        GetStringParam(policy->params, "gpu_multi_level_tiling_structure") :
        GetStringParam(policy->params, "cpu_multi_level_tiling_structure");

    std::vector<int> spatial_split_step_ids;
    int target_stage_id;
    std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;

    GetConsumers(task, state, state->stages[stage_id]->op, &consumers);
    CHECK(HasSingleElementwiseMatchedConsumer(task, state, stage, &target_stage_id));

    State base_state = state;
    base_state = DoMultiLevelTiling(base_state, stage_id,
        multi_level_tiling_structure, &spatial_split_step_ids);
    std::vector<int> follow_tiling_levels;
    if (IS_GPU(policy->cur_task)) {
      follow_tiling_levels.push_back(3);
    } else {
      follow_tiling_levels.push_back(1);
      follow_tiling_levels.push_back(2);
    }

    std::vector<std::pair<State, int> > ret;
    for (int level : follow_tiling_levels) {
      if (tolower(multi_level_tiling_structure[level-1]) != 's') {
        continue;
      }
      State tmp_s = base_state;
      tmp_s = FollowTiling(tmp_s, target_stage_id, spatial_split_step_ids, level);
      const Iterator &target_iter = tmp_s->stages[target_stage_id]->iters[
          level * spatial_split_step_ids.size() - 1];
      tmp_s.compute_at(stage_id, target_stage_id, target_iter);

      ret.emplace_back(std::move(tmp_s), stage_id - 1);
    }

    return ret;
  }
};

// The rule that adds a cache write stage
class RuleAddCacheWrite : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];

    int target_stage_id;

    // Add cache write if a stage needs multi-level tiling,
    // but does not have a element-wise matched consumer
    return NeedsMultilevelTiling(task, state, stage->op) &&
           !HasAttrsFlag(state, stage_id, SearchPolicyNode::no_cache_write_key) &&
           (!HasSingleElementwiseMatchedConsumer(task, state, stage,
                                                  &target_stage_id) ||
             (HasCacheReadStage(state, stage_id) &&
              !HasCacheWriteStage(state, stage_id))) ?
        kApply : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;

    State tmp_s = state;
    tmp_s.cache_write(stage_id, "local", task->compute_dag);
    return {std::make_pair(std::move(tmp_s), stage_id)};
  }
};

// The rule that adds a cache read stage
// Mainly used for GPU cooperative fetching
// Currently only support 1 to 1 match cache read
class RuleAddCacheRead : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    return ShouldBeCacheRead(policy, state, stage_id) ?
        kApplyAndSkipRest : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];

    std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;
    GetConsumers(task, state, stage->op, &consumers);
    CHECK_EQ(consumers.size(), 1);
    int target_stage_id = OperationToStage(*consumers.begin(), state);
    State tmp_s = state;
    int added_stage_id = tmp_s.cache_read(stage_id, "shared",
                                          {target_stage_id},
                                          task->compute_dag);
    target_stage_id++;
    const auto& share_read_pos = GetLastReduceIteratorInOutermostReduceTile(
        tmp_s->stages[target_stage_id]);
    tmp_s.compute_at(added_stage_id, target_stage_id, share_read_pos);

    return {std::make_pair(std::move(tmp_s), stage_id)};
  }
};

// The rule that adds rfactor stage
class RuleAddRfactor : public SketchGenerationRule {
 public:
  ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                              const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];

    return NeedsRfactor(task, state, stage->op) &&
           !HasCacheWriteStage(state, stage_id) ?
        kApply : kPass;
  }

  std::vector<std::pair<State, int> > Apply(const SketchSearchPolicyNode* policy,
                                            const State& state, int stage_id) final {
    const SearchTask& task = policy->cur_task;
    const Stage& stage = state->stages[stage_id];

    std::vector<std::pair<State, int> > ret;

    State tmp_s = state;

    // fuse reduce iters
    std::vector<Iterator> space_iters, reduce_iters;
    for (const auto &iter : stage->iters) {
      if (iter->iter_type == kSpace) {
        space_iters.push_back(iter);
      } else if (iter->iter_type == kReduce) {
        reduce_iters.push_back(iter);
      }
    }
    CHECK(!reduce_iters.empty());
    Iterator fused_reduce_iter;
    if (reduce_iters.size() > 1) {
      fused_reduce_iter = tmp_s.fuse(stage_id, reduce_iters);
    } else {
      fused_reduce_iter = reduce_iters[0];
    }

    // split reduce iters
    const auto &split_res = tmp_s.split(stage_id, fused_reduce_iter, {1});
    int factor_axis_id = static_cast<int>(space_iters.size());
    State base_state = tmp_s;
    for (const auto &split_iter : split_res) {
      tmp_s = base_state;
      tmp_s.rfactor(stage_id, split_iter, factor_axis_id, task->compute_dag);

      // reorder the space iterator to innermost for vectorization
      if (split_iter == split_res[1]) {
        std::vector<Iterator> new_order;
        for (size_t i = 0; i < tmp_s->stages[stage_id]->iters.size(); ++i) {
          if (i != space_iters.size()) {
            new_order.push_back(tmp_s->stages[stage_id]->iters[i]);
          }
        }
        new_order.push_back(tmp_s->stages[stage_id]->iters[space_iters.size()]);
        tmp_s.reorder(stage_id, new_order);
      }
      ret.emplace_back(std::move(tmp_s), stage_id - 1);
    }

    return ret;
  }
};

void SketchSearchPolicyNode::GenerateSketch(
    std::vector<State>* out_states) {
  State init_state = cur_task->compute_dag.GetInitState();
  std::string cpu_multi_level_tiling_structure =
      GetStringParam(params, "cpu_multi_level_tiling_structure");

  // two ping pong buffers to avoid copy
  std::vector<State> states_buf1, states_buf2;
  std::vector<State> *pnow, *pnext;
  pnow = &states_buf1;
  pnext = &states_buf2;
  pnow->push_back(init_state);

  // A map that maps state to its current working position (stage_id)
  std::unordered_map<State, int, ObjectHash, ObjectEqual> cur_stage_id_map;
  cur_stage_id_map[init_state] = static_cast<int>(init_state->stages.size() - 1);

  static RuleSkipStage rule_skip_stage;
  static RuleAlwaysInline rule_always_inline;
  static RuleMultiLevelTiling rule_multi_level_tiling;
  static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
  static RuleAddCacheWrite rule_add_cache_write_stage;
  static RuleAddCacheRead rule_add_cache_read_stage;
  static RuleAddRfactor rule_add_rfactor;
  if (sketch_rules.empty()) {
    // We may apply and skip the rest when processing some rules,
    // should take care of the rule vector order here
    sketch_rules.push_back(&rule_always_inline);
    sketch_rules.push_back(&rule_add_cache_write_stage);
    sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
    sketch_rules.push_back(&rule_multi_level_tiling);
    sketch_rules.push_back(&rule_add_rfactor);
    sketch_rules.push_back(&rule_skip_stage);
    if (IS_GPU(cur_task)) {
      // Try cache read first before cache write
      sketch_rules.insert(sketch_rules.begin() + 1, &rule_add_cache_read_stage);
    }
    // TODO(xian): Add a new rule to try combination of multi-level
    // tiling + rfactor
  }

  // Derivation rule based synthesizer
  while (!pnow->empty()) {
    pnext->clear();

    for (const State& state : *pnow) {
      int stage_id = cur_stage_id_map[state];

      // Reaches to the terminal stage
      if (stage_id < 0) {
        out_states->push_back(state);
        continue;
      }

      // Try all derivation rules
      for (const auto& rule : sketch_rules) {
        auto rule_check = rule->MeetCondition(this, state, stage_id);
        if (rule_check > SketchGenerationRule::ConditionEnum::kPass) {
          for (const auto& pair : rule->Apply(this, state, stage_id)) {
            cur_stage_id_map[pair.first] = pair.second;
            pnext->push_back(pair.first);
          }
          // Skip the reset rules
          if (rule_check == SketchGenerationRule::ConditionEnum::kApplyAndSkipRest) {
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
  for (size_t i = 0; i < out_states->size(); ++i) {
    auto pstate = (*out_states)[i].CopyOnWrite();
    for (size_t step_id = 0; step_id < pstate->transform_steps.size(); ++step_id) {
      if (pstate->transform_steps[step_id]->IsInstance<RfactorStepNode>()) {
        CHECK_GE(step_id, 1);
        int split_step_id = step_id - 1;
        auto step = pstate->transform_steps[split_step_id].as<SplitStepNode>();
        CHECK(step != nullptr);
        pstate->transform_steps[split_step_id]
            = SplitStep(step->stage_id, step->iter_id, step->extent, {PrimExpr()},
                        step->inner_to_outer);
      }
    }
  }

  StdCout(verbose) << "Generate Sketches\t\t#s: " << out_states->size() << std::endl;
}

int InitPopulationFillTileSize(const SketchSearchPolicyNode* policy,
                               State* state, std::mt19937* rand_gen,
                               SplitFactorizationMemo* split_memo) {
  for (size_t step_id = 0; step_id < (*state)->transform_steps.size(); ++step_id) {
    if (auto ps = (*state)->transform_steps[step_id].as<SplitStepNode>()) {
      bool defined = true;
      for (const PrimExpr& len : ps->lengths) {
        if (!len.defined()) {
          defined = false;
        }
      }

      if (defined) {
        continue;
      }

      int extent = GetIntImm(ps->extent);
      const std::vector<std::vector<PrimExpr> >& candidate_lens =
          split_memo->GetFactorizationSchemes(
              extent, ps->lengths.size(),
              policy->cur_task->hardware_params->max_innermost_split_factor);

      StateNode* pstate = state->CopyOnWrite();
      pstate->transform_steps[step_id] = SplitStep(
          ps->stage_id, ps->iter_id, ps->extent,
          candidate_lens[(*rand_gen)() % candidate_lens.size()],
          ps->inner_to_outer);
    }
  }

  return 0;
}

int InitPopulationThreadBind(const SketchSearchPolicyNode* policy,
                             State* state) {
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];
    auto pop = stage->op.as<te::ComputeOpNode>();

    if (stage->compute_at != kRoot || stage->op_type == kPlaceholder) {
      continue;
    }

    if (HasAnnotationIter(stage, IteratorAnnotation::kThreadX)) {
      // Skip if this stage has already done thread bind
      continue;
    }

    std::vector<Iterator> to_fuse;

    // This stage has not been tiled, but in GPU schedule, we must tile it
    // to do thread binding
    if (!HasSplitStep(*state, stage_id)) {
      for (const auto& it : (*state)->stages[stage_id]->iters) {
        if (it->iter_type == kReduce) {
          break;
        }
        to_fuse.push_back(it);
      }
      const auto& fused_it = state->fuse(stage_id, to_fuse);
      // Set default vthread=1 & threadIdx.x=default_warp_size
      // EvolutionarySearch will try more possiblity
      if (GetExtent(fused_it) <=
          policy->cur_task->hardware_params->warp_size) {
        state->bind_thread(stage_id, fused_it, kThreadX);
      } else {
        const auto& split_its = state->split(stage_id, fused_it,
            {1, policy->cur_task->hardware_params->warp_size});
        state->bind_thread(stage_id, split_its[0], kBlockX);
        state->bind_thread(stage_id, split_its[1], kVThread);
        state->bind_thread(stage_id, split_its[2], kThreadX);
      }

      continue;
    }

    int total_space_extent = 1;
    for (const auto& i : pop->root_iter_vars()) {
      CHECK(i->dom.defined());
      const auto& pint = i->dom->extent.as<IntImmNode>();
      CHECK(pint);
      total_space_extent *= pint->value;
    }

    // TODO(..): Add ThreadBind support for rfactor
    if (total_space_extent <= policy->cur_task->hardware_params->warp_size) {
      for (const auto& it : (*state)->stages[stage_id]->iters) {
        if (it->iter_type == kReduce) {
          break;
        }
        to_fuse.push_back(it);
      }
      const auto& fused_it = state->fuse(stage_id, to_fuse);
      state->bind_thread(stage_id, fused_it, kThreadX);

      continue;
    }

    // Fuse the outermost space tile as blockIdx
    for (size_t i = 0; i < pop->axis.size(); i++) {
      const auto& it = (*state)->stages[stage_id]->iters[i];
      if (!StrEndsWith(it->name, ".0")) {
        break;
      }
      to_fuse.push_back(it);
    }
    const auto& blockidx_it = state->fuse(stage_id, to_fuse);
    state->bind_thread(stage_id, blockidx_it, kBlockX);

    // Fuse the second outermost space tile as vthread
    to_fuse.clear();
    for (size_t i = 1; i < pop->axis.size() + 1; i++) {
      const auto& it = (*state)->stages[stage_id]->iters[i];
      if (!StrEndsWith(it->name, ".1")) {
        break;
      }
      to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
    }
    const auto& vthread_it = state->fuse(stage_id, to_fuse);
    if (GetExtent(vthread_it) >
        policy->cur_task->hardware_params->max_vthread_extent) {
      return -1;
    }
    state->bind_thread(stage_id, vthread_it, kVThread);

    // Fuse the third outermost space tile as threadIdx
    to_fuse.clear();
    for (size_t i = 2; i < pop->axis.size() + 2; i++) {
      const auto& it = (*state)->stages[stage_id]->iters[i];
      if (!StrEndsWith(it->name, ".2")) {
        break;
      }
      to_fuse.push_back((*state)->stages[stage_id]->iters[i]);
    }
    const auto& threadidx_it = state->fuse(stage_id, to_fuse);
    if (GetExtent(threadidx_it) <
        policy->cur_task->hardware_params->warp_size) {
      return -1;
    }
    state->bind_thread(stage_id, threadidx_it, kThreadX);
  }

  return 0;
}

int InitPopulationCooperativeFetching(const SketchSearchPolicyNode* policy,
                                      State* state) {
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    // Do cooperative fetching with cache read stage
    // For two stages: A -> B
    // 1. A -> A_cache_read -> B
    //               *
    // 2. A -> A_cache_write -> A_cache_read -> B
    //                                *
    if ((stage_id > 0 && HasCacheReadStage((*state), stage_id - 1) &&
         !HasCacheWriteStage((*state), stage_id - 1)) ||
        (stage_id > 1 && HasCacheReadStage((*state), stage_id - 2) &&
         HasCacheWriteStage((*state), stage_id - 2))) {
      const Stage& target_stage = (*state)->stages[stage_id];
      if (HasAnnotationIter(target_stage, IteratorAnnotation::kThreadX) ||
          HasAnnotationIter(target_stage, IteratorAnnotation::kTensorized)) {
        // Skip if this stage has already done thread bind or has been
        // tensorized
        continue;
      }
      // Get spatial_split_step_ids from the root stage
      std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;
      std::vector<int> spatial_split_step_ids;
      GetConsumers(policy->cur_task, (*state), target_stage->op, &consumers);
      CHECK_EQ(consumers.size(), 1);
      int target_stage_id = OperationToStage(*consumers.begin(), (*state));
      GetSpaceSplitStepIds((*state), target_stage_id, &spatial_split_step_ids);

      // Fuse all axis to to do cooperative fetching
      Iterator fused = state->fuse(stage_id,
                                   (*state)->stages[stage_id]->iters);
      // Left a vectorized cooperative fetching split placeholder
      const auto& iters0 = state->split(stage_id, fused, {1});
      state->vectorize(stage_id, iters0[1]);
      // Follow split to keep a same thread extent with the root stage
      const auto& iters1 = state->follow_fused_split(stage_id, iters0[0],
                                                     spatial_split_step_ids,
                                                     1, true);
      state->bind_thread(stage_id, iters1[1], kThreadX);
    }
  }

  return 0;
}

int InitPopulationChangeComputeLocation(const SketchSearchPolicyNode* policy,
                                        State* state, std::mt19937* rand_gen) {
  if(GetIntParam(policy->params, "disable_change_compute_location")) {
    return 0;
  }

  for (int stage_id = static_cast<int>((*state)->stages.size()) - 1; stage_id >= 0; stage_id--) {
    const Stage& stage = (*state)->stages[stage_id];

    if (stage->op_type == kPlaceholder) {
      continue;
    }

    if (IsTiled(stage) || stage->compute_at == kInlined) {
      continue;
    }

    if (NeedsMultilevelTiling(policy->cur_task, (*state), stage->op)) {
      continue;
    }

    std::unordered_set<te::Operation, ObjectHash, ObjectEqual> consumers;

    GetConsumers(policy->cur_task, (*state), stage->op, &consumers);
    if (consumers.empty()) {
      continue;
    }

    int target_stage_id;
    if (consumers.size() == 1) {
      target_stage_id = OperationToStage(*consumers.begin(), *state);
    } else {
      // check all consumers share a common root
      int common_root_id = -1;
      bool mismatch = false;
      for (const auto& consumer : consumers) {
        int consumer_stage_id = OperationToStage(consumer, *state);
        int root_id = -1;
        if ((*state)->stages[consumer_stage_id]->compute_at == kRoot) {
          root_id = consumer_stage_id;
        } else if ((*state)->stages[consumer_stage_id]->compute_at == kIter) {
          root_id = (*state)->attach_map->stage_to_attach_iter.at(consumer_stage_id).first;
        } else {
          LOG(FATAL) << "Invalid case";
        }

        if (common_root_id == -1) {
          common_root_id = root_id;
        } else {
          if (common_root_id != root_id) {
            mismatch = true;
            break;
          }
        }
      }

      if (mismatch) {
        continue;
      }
      target_stage_id = common_root_id;
    }

    const Stage& target_stage = (*state)->stages[target_stage_id];
    std::set<std::string> to_unroll_name_set;
    if (target_stage->op->attrs.count(policy->always_unroll_key)) {
      to_unroll_name_set = GetIterNameSetParam(target_stage->op->attrs,
                                               policy->always_unroll_key);
    }

    std::vector<std::pair<int, Iterator> > candidates;
    bool target_compute_at_other = target_stage->compute_at == kIter;
    bool target_is_tiled = IsTiled(target_stage);

    bool visited_reduce = false;
    // enumerate compute_at location at target_stage
    int ct = 0;
    for (const auto& target_iter : target_stage->iters) {
      if (target_iter->iter_type == kReduce) {
        visited_reduce = true;
        if (!target_is_tiled) {  // do not go into reduce iter
          break;
        }
      } else if (target_iter->iter_type == kSpace) {
        if (visited_reduce) {  // do not go into inner tile
          break;
        }
      }

      if (to_unroll_name_set.count(target_iter->name)) {
          // Do not go into always unroll region
          break;
      }

      if (GetExtent(target_iter) == 1) {  // skip iterators with length of 1
        continue;
      }
      if (target_compute_at_other && target_iter->iter_type == kSpace &&
          StrEndsWith(target_iter->name, ".0")) {
        // skip the first level iterators if target stage compute_at another stage
        // In this case, the lengths of first level iterators are always one
        continue;
      }
      candidates.emplace_back(target_stage_id, target_iter);

      if ((*state)->attach_map->iter_to_attached_stages.count(
          std::make_pair(target_stage_id, ct++))) {
        break;
      }
    }

    // if the target_stage is already compute_at another stage X, try also compute_at X
    // We call stage X as `target_target_stage`
    if (target_compute_at_other) {
      int target_target_stage_id;
      target_target_stage_id = (*state)->attach_map->stage_to_attach_iter.at(
          target_stage_id).first;
      const Stage& target_target_stage = (*state)->stages[target_target_stage_id];
      if (target_target_stage->op->attrs.count(policy->always_unroll_key)) {
        to_unroll_name_set = GetIterNameSetParam(target_target_stage->op->attrs,
                                                 policy->always_unroll_key);
      } else {
        to_unroll_name_set.clear();
      }

      int ct = 0;
      for (const auto& target_target_iter : target_target_stage->iters) {
        if (target_target_iter->iter_type == kReduce ||
            (*state)->attach_map->iter_to_attached_stages.count(
                std::make_pair(target_target_stage_id, ct++))) {
          break;
        }

        if (to_unroll_name_set.count(target_target_iter->name)) {
            // Do not go into always unroll region
            break;
        }

        if (GetExtent(target_target_iter) == 1) {  // skip iterators with length of 1
          continue;
        }

        candidates.push_back(std::make_pair(target_target_stage_id, target_target_iter));
      }
    }

    int choice = (*rand_gen)() % (candidates.size() + 2);

    if (choice == 0) {
      if (!HasReduceIter(stage)) {
        state->compute_inline(stage_id);
      }
    } else if (choice == 1) {
      state->compute_root(stage_id);
    } else {
      choice = choice - 2;
      state->compute_at(stage_id, candidates[choice].first, candidates[choice].second);
    }
  }

  return 0;
}

int InitPopulationParallel(const SketchSearchPolicyNode* policy,
                           State* state) {
  std::function<void(const SketchSearchPolicyNode*, State*, int stage_id, int iter_offset)> annotate_parallel;

  annotate_parallel = [&annotate_parallel](
          const SketchSearchPolicyNode* policy, State* state, int stage_id, int iter_offset) {
    const Stage& stage = (*state)->stages[stage_id];

    std::vector<Iterator> to_fuse;
    int64_t parallel_degree = 1;

    // strategy: try to fuse and parallel the outermost n iterators
    // Stop if we meet reduce iterator or we have enough parallel degree
    size_t iter_id = iter_offset;
    for (; iter_id < stage->iters.size(); ++iter_id) {
      const Iterator& it = stage->iters[iter_id];
      if (it->iter_type == kReduce || it->annotation != kNone) {
        break;
      }

      to_fuse.push_back(it);
      parallel_degree *= GetExtent(it);

      if (parallel_degree > policy->cur_task->hardware_params->num_cores * 16) {
        break;
      }

      if ((*state)->attach_map->iter_to_attached_stages.count(
          std::make_pair(stage_id, iter_id))) {
        break;
      }
    }

    if (parallel_degree == 1) {
      auto res = (*state)->attach_map->iter_to_attached_stages.find(std::make_pair(stage_id, iter_id));
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
    if (stage->compute_at != kRoot || stage->op_type == kPlaceholder) {
      continue;
    }

    annotate_parallel(policy, state, stage_id, 0);
  }

  return 0;
}

int InitPopulationVectorization(const SketchSearchPolicyNode* policy,
                                State* state, std::mt19937* rand_gen) {
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];

    if (stage->op_type == kPlaceholder) {
      continue;
    }

    // Skip cooperative fetching stage
    if (IS_GPU(policy->cur_task) &&
        HasCacheReadStage((*state), stage_id - 1)) {
      continue;
    }

    if (HasAnnotationIter(stage, IteratorAnnotation::kTensorized)) {
      // Skip if this stage has been tensorized
      continue;
    }

    // try to fuse and vectorize the space iterators in the inner most tile
    int cum_length_prod = 1;

    std::set<std::string> to_unroll_name_set;
    if (stage->op->attrs.count(policy->always_unroll_key)) {
      to_unroll_name_set = GetIterNameSetParam(stage->op->attrs,
                                               policy->always_unroll_key);
    }

    int num_fusible = 0;
    while (num_fusible < static_cast<int>(stage->iters.size())) {
      int iter_id = static_cast<int>(stage->iters.size()) - 1 - num_fusible;
      if ((*state)->attach_map->iter_to_attached_stages.count(
          std::make_pair(stage_id, iter_id))) {
        break;
      }

      const Iterator& it = stage->iters[iter_id];

      // Stop if we meet a reduce iterator
      if (it->iter_type == kReduce || it->annotation != kNone ||
          to_unroll_name_set.count(it->name)) {
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
      if (cum_length_prod > policy->cur_task->hardware_params->max_unroll_vec) {
        break;
      }

      num_fusible++;
    }

    if (num_fusible > 1) {
      num_fusible = 1 + (*rand_gen)() % (num_fusible - 1); // Select a random range to fuse
    }

    if (num_fusible == 1) {
      state->vectorize(stage_id, stage->iters.back());
    } else if (num_fusible > 1) {
      std::vector<Iterator> to_fuse(stage->iters.end() - num_fusible,
                                    stage->iters.end());
      state->vectorize(stage_id, state->fuse(stage_id, to_fuse));
    }
  }

  return 0;
}

int InitPopulationUnroll(const SketchSearchPolicyNode* policy,
                         State* state, std::mt19937* rand_gen) {
  for (size_t stage_id = 0; stage_id < (*state)->stages.size(); ++stage_id) {
    const Stage& stage = (*state)->stages[stage_id];

    if (stage->op_type == kPlaceholder) {
      continue;
    }

    if (stage->op->attrs.count(policy->always_unroll_inner_key)) {
      // Special unroll policy
      auto to_unroll_name_set = GetIterNameSetParam(stage->op->attrs,
                                                    policy->always_unroll_inner_key);
      std::set<std::string> visited_names;

      // Unroll the space iterators and reduce iterators listed in the attrs
      // in the innermost tile
      int n = static_cast<int>(stage->iters.size()) - 1;
      visited_names.clear();
      while (n >= 0) {
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
          state->unroll(stage_id, it);
        }

        n--;
      }
    } else if (stage->op->attrs.count(policy->always_unroll_key)) {
      // Special unroll policy
      auto to_unroll_name_set = GetIterNameSetParam(stage->op->attrs,
                                                    policy->always_unroll_key);

      // Unroll the space iterators and reduce iterators listed in the attrs
      int n = static_cast<int>(stage->iters.size()) - 1;
      while (n >= 0) {
        const Iterator& it = stage->iters[n];
        if (to_unroll_name_set.count(it->name)) {
          state->unroll(stage_id, it);
        }
        n--;
      }
    } else if (HasReduceIter(stage)) {
      // use auto unroll for multi level tiled stage
      int value = policy->auto_unroll_configs[
          (*rand_gen)() % policy->auto_unroll_configs.size()];
      state->pragma(stage_id, (*state)->stages[stage_id]->iters[0],
                    std::string("auto_unroll_max_step") + "$" + std::to_string(value));
    }
  }

  return 0;
}

void SketchSearchPolicyNode::SampleInitPopulation(const std::vector<State>& sketches,
    int out_size, std::vector<State>* out_states) {
  std::uniform_real_distribution<> dis(0.0, 1.0);
  int continue_count = 0;

  // TODO(...): Maybe try muti thread here
  while (static_cast<int>(out_states->size()) < out_size &&
         continue_count < out_size * 10) {
    State tmp_s = sketches[rand_gen_() % sketches.size()];

    InitPopulationFillTileSize(this, &tmp_s, &rand_gen_, &split_memo_);

    if (IS_GPU(cur_task)) {
      tmp_s = cur_task->compute_dag.InferBound(tmp_s);

      if (InitPopulationThreadBind(this, &tmp_s)) {
        continue_count++;
        if (continue_count == out_size) {
          StdCout(verbose) << "Initial Population Sampling..." << std::endl;
        }
        continue;
      }

      InitPopulationCooperativeFetching(this, &tmp_s);
    } else {
      InitPopulationChangeComputeLocation(this, &tmp_s, &rand_gen_);

      tmp_s = cur_task->compute_dag.InferBound(tmp_s);

      InitPopulationParallel(this, &tmp_s);
    }

    InitPopulationVectorization(this, &tmp_s, &rand_gen_);

    InitPopulationUnroll(this, &tmp_s, &rand_gen_);

    out_states->push_back(std::move(tmp_s));
  }

  StdCout(verbose) << "Sample Initial Population\t#s: "
                   << out_states->size() << std::endl;
}

void SketchSearchPolicyNode::EvolutionarySearch(
    const std::vector<State>& init_population,
    int num_best_states, std::vector<State>* best_states) {
  auto tic_begin = std::chrono::high_resolution_clock::now();

  // Set parameters for genetic algorithm
  int population = GetIntParam(params, "evolutionary_search_population");
  int num_iters =  GetIntParam(params, "evolutionary_search_num_iters");
  double mutation_prob = GetDoubleParam(params, "evolutionary_search_mutation_prob");
  int num_cross_over = static_cast<int>(population * 0.0);  // NOT IMPLEMENTED currently
  int num_cross_over_trial_upper_bound = num_cross_over * 3;
  CostModel cost_model = program_cost_model;

  // Two ping pong buffers to avoid copy
  std::vector<State> states_buf1, states_buf2;
  std::vector<State> *pnow = &states_buf1, *pnext = &states_buf2;
  states_buf1.reserve(population);
  states_buf2.reserve(population);
  states_buf1.insert(states_buf1.begin(), init_population.begin(), init_population.end());

  // A heap to keep the best states during evolution
  using StateItem = std::pair<State, float>;
  auto cmp = [](const StateItem& left, const StateItem& right) {
    return left.second > right.second;
  };
  std::vector<StateItem> heap;
  std::unordered_set<std::string> in_heap(measured_states_set_);
  heap.reserve(num_best_states);

  // auxiliary global variables
  std::vector<float> scores;
  std::vector<double> prefix_sum_probs;
  double max_score = 0.0;
  scores.reserve(population);
  prefix_sum_probs.reserve(population);
  std::uniform_real_distribution<> dis(0.0, 1.0);
  int mutation_fail_ct = 0;

  // Genetic Algorithm
  for (int k = 0; k < num_iters + 1; ++k) {
    // Maintain the heap
    cur_task->compute_dag.InferBound(pnow);
    PruneUndefined(pnow);
    cost_model->Predict(cur_task, *pnow, &scores);

    for (size_t i = 0; i < pnow->size(); ++i) {
      const State& state = (*pnow)[i];
      std::string state_str = state.ToStr();

      if (in_heap.count(state_str) == 0) {
        if (static_cast<int>(heap.size()) < num_best_states) {
          heap.emplace_back((*pnow)[i], scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
          in_heap.insert(state_str);
        } else if (scores[i] > heap.front().second) {
          std::string old_state_str = heap.front().first.ToStr();
          in_heap.erase(old_state_str);
          in_heap.insert(state_str);

          std::pop_heap(heap.begin(), heap.end(), cmp);
          heap.back() = StateItem(state, scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
        }
        if (scores[i] > max_score) {
          max_score = scores[i];
        }
      }
    }

    if (k % 5 == 0 || k == num_iters) {
      StdCout(verbose) << "GA Iter: " << k << std::fixed << std::setprecision(4)
                       << "\tMax score: " << max_score
                       << "\tMin score: " << heap.front().second
                       << "\tPop size: " << pnow->size() << std::endl;
    }

    if (k == num_iters) {
      break;
    }

    // Compute selection probability
    double sum = 0.0;
    prefix_sum_probs.resize(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
      sum += std::max(scores[i], 0.0f);
      prefix_sum_probs[i] = sum;
    }
    for (size_t i = 0; i < scores.size(); ++i) {
      prefix_sum_probs[i] = prefix_sum_probs[i] / sum;
    }

    // Do cross over
    int ct = 0;
    while (static_cast<int>(pnext->size()) < num_cross_over
        && ct < num_cross_over_trial_upper_bound) {
      int p1 = RandomChoose(prefix_sum_probs, &rand_gen_);
      int p2 = RandomChoose(prefix_sum_probs, &rand_gen_);

      if (p1 == p2) {
        pnext->push_back((*pnow)[p1]);
      } else {
        State tmp_s = CrossOverState((*pnow)[p1], (*pnow)[p2]);
        if (tmp_s.defined()) {
          pnext->push_back(std::move(tmp_s));
        }
      }
      ct++;
    }

    // Do mutation
    mutation_fail_ct = 0;
    while (static_cast<int>(pnext->size()) < population) {
      int id = RandomChoose(prefix_sum_probs, &rand_gen_);

      if (dis(rand_gen_) < mutation_prob) {
        const std::vector<double> rule_prefix_sum_probs{0.9, 1.0};

        int rule_id = RandomChoose(rule_prefix_sum_probs, &rand_gen_);

        if (rule_id == 0) {
          // Mutate Tile Size
          State tmp_s = RandomMutateTileSize((*pnow)[id], &split_memo_, &rand_gen_,
                                             cur_task->hardware_params->max_innermost_split_factor);
          if (tmp_s.defined()) {
            pnext->push_back(std::move(tmp_s));
          } else {
            mutation_fail_ct++;
          }
        } else if (rule_id == 1) {
          // Mutate auto-unroll max step.
          State tmp_s = RandomMutateMaxUnrollStep((*pnow)[id], &rand_gen_, auto_unroll_configs);
          if (tmp_s.defined()) {
            pnext->push_back(std::move(tmp_s));
          } else {
            mutation_fail_ct++;
          }
        }
      } else {
        pnext->push_back((*pnow)[id]);
      }
    }

    std::swap(pnext, pnow); pnext->clear();
  }

  // Copy best states in the heap to out_states
  std::sort(heap.begin(), heap.end(), cmp);
  best_states->clear();
  for (auto& item : heap) {
    best_states->push_back(std::move(item.first));
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double> >(
      std::chrono::high_resolution_clock::now()-  tic_begin).count();
  StdCout(verbose) << "EvolutionarySearch\t\t#s: " << best_states->size()
                   << "\tTime elapsed: "
                   << std::fixed << std::setprecision(2) << duration << std::endl;
}

class RuleCustomSketch : public SketchGenerationRule {
 public:
  RuleCustomSketch(PackedFunc meet_condition_func, PackedFunc apply_func) :
      meet_condition_func_(meet_condition_func), apply_func_(apply_func) {}

  inline ConditionEnum MeetCondition(const SketchSearchPolicyNode* policy,
                                     const State& state, int stage_id) final {
    auto ret = meet_condition_func_(
        tvm::runtime::GetRef<SketchSearchPolicy>(policy), state, stage_id);
    if (ret.type_code() == 0) {
      return ConditionEnum(static_cast<int>(ret));
    } else {
      return kApplyAndSkipRest;
    }
  }

  inline std::vector<std::pair<State, int> > Apply(
      const SketchSearchPolicyNode* policy,
      const State& state, int stage_id) final {
    std::vector<std::pair<State, int> > ret;

    Array<Array<ObjectRef>> apply_ret = apply_func_(
        tvm::runtime::GetRef<SketchSearchPolicy>(policy), state, stage_id);

    for (const auto& item : apply_ret) {
      CHECK_EQ(item.size(), 2);
      State state = Downcast<State>(item[0]);
      auto next = item[1].as<IntImmNode>();
      ret.emplace_back(state, next->value);
    }
    return ret;
  }

 private:
  PackedFunc meet_condition_func_;
  PackedFunc apply_func_;
};

PreloadCustomSketchRule::PreloadCustomSketchRule(PackedFunc meet_condition_func,
                                                 PackedFunc apply_func) {
  auto node = make_object<PreloadCustomSketchRuleNode>();
  node->meet_condition_func = meet_condition_func;
  node->apply_func = apply_func;
  data_ = std::move(node);
}

void PreloadCustomSketchRuleNode::callback(SearchPolicyNode* policy) {
  CHECK(policy->IsInstance<SketchSearchPolicyNode>());
  auto sketch_policy = dynamic_cast<SketchSearchPolicyNode*>(policy);
  sketch_policy->sketch_rules.emplace_back(
      new RuleCustomSketch(meet_condition_func, apply_func));
  StdCout(policy->verbose) << "Custom sketch rule added." << std::endl;
}

TVM_REGISTER_GLOBAL("ansor.SketchSearchPolicy")
.set_body_typed([](CostModel program_cost_model, Map<String, ObjectRef> params,
                   int seed){
  return SketchSearchPolicy(program_cost_model, params, seed);
});

TVM_REGISTER_GLOBAL("ansor.PreloadCustomSketchRule")
.set_body_typed([](PackedFunc meet_condition_func, PackedFunc apply_func) {
  return PreloadCustomSketchRule(meet_condition_func, apply_func);
});

}  // namespace ansor
}  // namespace tvm
