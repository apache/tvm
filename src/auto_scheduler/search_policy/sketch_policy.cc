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

#include "sketch_policy.h"

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

#include "sketch_policy_rules.h"

namespace tvm {
namespace auto_scheduler {

/********** Sketch generation rules **********/

static RuleSkipStage rule_skip_stage;
static RuleAlwaysInline rule_always_inline;
static RuleMultiLevelTiling rule_multi_level_tiling;
static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
static RuleAddCacheWrite rule_add_cache_write_stage;
static RuleAddRfactor rule_add_rfactor;
static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;

/********** Init population rules **********/

static InitFillTileSize init_fill_tile_size;
static InitChangeComputeLocation init_change_compute_location;
static InitParallel init_parallel;
static InitUnroll init_unroll;
static InitVectorization init_vectorization;

/********** Sketch policy **********/

TVM_REGISTER_NODE_TYPE(SketchPolicyNode);

SketchPolicy::SketchPolicy(SearchTask task, CostModel schedule_cost_model,
                           Map<String, ObjectRef> params, int seed, int verbose,
                           Optional<Array<SearchCallback>> init_search_callbacks) {
  auto node = make_object<SketchPolicyNode>();
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

State SketchPolicyNode::Search(int n_trials, int early_stopping, int num_measure_per_iter,
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

Array<State> SketchPolicyNode::SearchOneRound(int num_random_states, Array<State>* random_states) {
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

Array<State> SketchPolicyNode::GenerateSketches() {
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

Array<State> SketchPolicyNode::SampleInitPopulation(const Array<State>& sketches, int out_size) {
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

Array<State> SketchPolicyNode::EvolutionarySearch(const Array<State>& init_populations,
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

Array<MeasureInput> SketchPolicyNode::PickStatesWithEpsGreedy(const Array<State>& best_states,
                                                              const Array<State>& random_states,
                                                              int remaining_n_trials) {
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

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicy")
    .set_body_typed([](SearchTask task, CostModel schedule_cost_model,
                       Map<String, ObjectRef> params, int seed, int verbose,
                       Optional<Array<SearchCallback>> init_search_callbacks) {
      return SketchPolicy(task, schedule_cost_model, params, seed, verbose, init_search_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicyGenerateSketches")
    .set_body_typed([](SketchPolicy policy) { return policy->GenerateSketches(); });

}  // namespace auto_scheduler
}  // namespace tvm
