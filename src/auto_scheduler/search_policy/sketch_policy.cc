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
#include <tvm/support/parallel_for.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
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
static RuleAddCacheRead rule_add_cache_read_stage;
static RuleAddCacheWrite rule_add_cache_write_stage;
static RuleAddRfactor rule_add_rfactor;
static RuleCrossThreadReduction rule_cross_thread_reduction;
static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;
static RuleSpecialComputeLocationGPU rule_special_compute_location_gpu;

/********** Init population rules **********/
static InitFillTileSize init_fill_tile_size;
static InitChangeComputeLocation init_change_compute_location;
static InitParallel init_parallel;
static InitUnroll init_unroll;
static InitVectorization init_vectorization;
static InitThreadBind init_thread_bind;

/********** Sketch policy **********/
TVM_REGISTER_NODE_TYPE(SketchPolicyNode);

SketchPolicy::SketchPolicy(SearchTask task, CostModel program_cost_model,
                           Map<String, ObjectRef> params, int seed, int verbose,
                           Optional<Array<SearchCallback>> init_search_callbacks) {
  auto node = make_object<SketchPolicyNode>();
  node->search_task = std::move(task);
  node->program_cost_model = std::move(program_cost_model);
  node->rand_gen = std::mt19937(seed);
  node->params = std::move(params);
  node->verbose = verbose;
  node->sample_init_min_pop_ =
      GetIntParam(node->params, SketchParamKey::SampleInitPopulation::min_population);

  if (init_search_callbacks) {
    PrintTitle("Call init-search callbacks", verbose);
    // Candidates:
    // - auto_scheduler.PreloadMeasuredStates: Load already measured states to
    //   `measured_states_set_`, `measured_states_vector_` and `measured_states_throughputs_`.
    // - auto_scheduler.PreloadCustomSketchRule: Add user custom sketch rules to `sketch_rules`,
    //   these rules will be processed prior to the default rules.
    node->RunCallbacks(init_search_callbacks.value());
  }

  // NOTE: There are strong dependency among the rules below,
  // so the order to push them into the vector should be considered carefully.
  if (IsCPUTask(node->search_task)) {
    // Sketch Generation Rules
    node->sketch_rules.push_back(&rule_always_inline);
    node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
    node->sketch_rules.push_back(&rule_add_rfactor);
    node->sketch_rules.push_back(&rule_add_cache_write_stage);
    node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
    node->sketch_rules.push_back(&rule_multi_level_tiling);
    node->sketch_rules.push_back(&rule_skip_stage);

    // Initial Population Generation Rules
    node->init_rules.push_back(&init_fill_tile_size);
    node->init_rules.push_back(&init_change_compute_location);
    node->init_rules.push_back(&init_parallel);
    node->init_rules.push_back(&init_unroll);
    node->init_rules.push_back(&init_vectorization);

    // Mutation Rules for Evolutionary Search
    node->mutation_rules.push_back(std::make_shared<MutateTileSize>(0.90));
    node->mutation_rules.push_back(std::make_shared<MutateAutoUnroll>(0.04));
    node->mutation_rules.push_back(std::make_shared<MutateComputeLocation>(0.05));
    node->mutation_rules.push_back(std::make_shared<MutateParallel>(0.01));
  } else if (IsGPUTask(node->search_task)) {
    // Sketch Generation Rules
    if (node->search_task->target->GetAttr<String>("device", "") == "mali") {
      node->sketch_rules.push_back(&rule_always_inline);
      node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      node->sketch_rules.push_back(&rule_add_rfactor);
      node->sketch_rules.push_back(&rule_add_cache_write_stage);
      node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      node->sketch_rules.push_back(&rule_multi_level_tiling);
      node->sketch_rules.push_back(&rule_skip_stage);
    } else {
      node->sketch_rules.push_back(&rule_add_cache_read_stage);
      node->sketch_rules.push_back(&rule_special_compute_location_gpu);
      node->sketch_rules.push_back(&rule_always_inline);
      node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      node->sketch_rules.push_back(&rule_cross_thread_reduction);
      node->sketch_rules.push_back(&rule_add_cache_write_stage);
      node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      node->sketch_rules.push_back(&rule_multi_level_tiling);
      node->sketch_rules.push_back(&rule_skip_stage);
    }

    // Initial Population Generation Rules
    node->init_rules.push_back(&init_fill_tile_size);
    node->init_rules.push_back(&init_thread_bind);
    node->init_rules.push_back(&init_unroll);

    if (node->search_task->target->GetAttr<String>("device", "") == "mali") {
      node->init_rules.push_back(&init_vectorization);
    }

    // Mutation Rules for Evolutionary Search
    node->mutation_rules.push_back(std::make_shared<MutateTileSize>(0.90));
    node->mutation_rules.push_back(std::make_shared<MutateAutoUnroll>(0.10));
  } else {
    LOG(FATAL) << "No default sketch rules for target: " << node->search_task->target;
  }

  data_ = std::move(node);
}

State SketchPolicyNode::Search(int n_trials, int early_stopping, int num_measure_per_iter,
                               ProgramMeasurer measurer) {
  num_measure_per_iter_ = num_measure_per_iter;

  if (n_trials <= 1) {
    // No measurement is allowed
    const Array<State>& best_states = SearchOneRound(0);
    ICHECK_GT(best_states.size(), 0);
    return best_states[0];
  } else {
    int num_random =
        static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy) * num_measure_per_iter);
    early_stopping = early_stopping < 0 ? std::numeric_limits<int>::max() >> 1 : early_stopping;
    measurer->Reset();

    int ct = 0;
    int empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);
    Array<State> best_states, random_states;
    Array<MeasureInput> inputs;
    Array<MeasureResult> results;
    while (ct < n_trials) {
      if (!inputs.empty()) {
        auto t_begin = std::chrono::high_resolution_clock::now();

        // Retrain the cost model before the next search round
        PrintTitle("Train cost model", verbose);
        program_cost_model->Update(inputs, results);

        PrintTimeElapsed(t_begin, "training", verbose);
      }

      // Search one round to get promising states
      PrintTitle("Search", verbose);
      best_states = SearchOneRound(num_random * 3, &random_states);

      // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
      best_states = search_task->compute_dag.InferBound(best_states);
      random_states = search_task->compute_dag.InferBound(random_states);

      // Pick `num_measure_per_iter` states to measure, check hash to remove already measured state
      // Also pick some random states to do eps-greedy
      inputs = PickStatesWithEpsGreedy(best_states, random_states, n_trials - ct);

      // Currently it's hard to detect if all of the search space has been traversed
      // Stop if no extra valid states found in several retries
      if (inputs.empty()) {
        if (empty_retry_count-- > 0) {
          continue;
        } else {
          StdCout(verbose) << "It seems all candidates in the search space have been measured."
                           << std::endl;
          break;
        }
      } else {
        // Reset the retry count
        empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);
      }

      // Measure candidate states
      PrintTitle("Measure", verbose);
      results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);
      ct += inputs.size();

      // Check if reach the early stopping condition
      if (ct - measurer->best_ct[search_task->workload_key] > early_stopping &&
          measurer->has_valid.count(search_task->workload_key)) {
        StdCout(verbose) << "Stop early since no performance improvement in the last "
                         << early_stopping << " measurements trials.\n";
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

std::pair<Array<MeasureInput>, Array<MeasureResult>> SketchPolicyNode::ContinueSearchOneRound(
    int num_measure, ProgramMeasurer measurer) {
  num_measure_per_iter_ = num_measure;

  Array<State> best_states, random_states;
  Array<MeasureInput> inputs;
  Array<MeasureResult> results;
  int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure);

  // Search one round to get promising states
  PrintTitle("Search", verbose);
  best_states = SearchOneRound(num_random * 3, &random_states);

  // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
  best_states = search_task->compute_dag.InferBound(best_states);
  random_states = search_task->compute_dag.InferBound(random_states);

  // Pick `num_measure_per_iter` states to measure, check hash to remove already measured state
  // Also pick some random states to do eps-greedy
  inputs = PickStatesWithEpsGreedy(best_states, random_states, num_measure);

  // Measure candidate states
  PrintTitle("Measure", verbose);
  results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);

  // Update measured states throughputs. These states will join the EvolutionarySearch in later
  // search rounds.
  for (const auto& res : results) {
    measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
  }

  auto t_begin = std::chrono::high_resolution_clock::now();

  // Update the cost model
  PrintTitle("Train cost model", verbose);
  program_cost_model->Update(inputs, results);

  PrintTimeElapsed(t_begin, "training", verbose);

  return std::make_pair(std::move(inputs), std::move(results));
}

Array<State> SketchPolicyNode::SearchOneRound(int num_random_states, Array<State>* random_states) {
  // Get parameters
  int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
  int num_use_measured = std::min(
      static_cast<int>(measured_states_vector_.size()),
      static_cast<int>(
          GetDoubleParam(params, SketchParamKey::SampleInitPopulation::use_measured_ratio) *
          population));

  // 1. Generate sketches
  if (sketch_cache_.empty()) {
    sketch_cache_ = GenerateSketches();
  }

  // 2. Sample the init population
  Array<State> init_population = SampleInitPopulation(sketch_cache_);

  // 3. Perform evolutionary search.
  // Also insert already measured good states to the initial population
  std::vector<int> indices = Argsort(measured_states_throughputs_);
  for (int i = 0; i < num_use_measured; i++) {
    init_population.push_back(measured_states_vector_[indices[i]]);
  }
  // Sample some random states for eps-greedy
  if (num_random_states > 0 && random_states != nullptr) {
    *random_states = RandomSampleStates(init_population, &rand_gen, num_random_states);
  }
  return EvolutionarySearch(init_population, num_measure_per_iter_ * 2);
}

Array<State> SketchPolicyNode::GenerateSketches() {
  const State& init_state = search_task->compute_dag->init_state;

  // Two ping pong buffers to avoid copy
  Array<State> states_buf1{init_state}, states_buf2;
  Array<State>* pnow = &states_buf1;
  Array<State>* pnext = &states_buf2;

  // A map that maps state to its current working position (stage_id)
  std::unordered_map<State, int, ObjectHash, ObjectEqual> cur_stage_id_map;
  cur_stage_id_map[init_state] = static_cast<int>(init_state->stages.size()) - 1;

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
          // Skip the rest rules
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
        ICHECK_GE(step_id, 1);
        int split_step_id = static_cast<int>(step_id - 1);
        auto step = pstate->transform_steps[split_step_id].as<SplitStepNode>();
        ICHECK(step != nullptr);
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

Array<State> SketchPolicyNode::SampleInitPopulation(const Array<State>& sketches) {
  // Use this population as the parallel degree to do sampling
  int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);

  auto tic_begin = std::chrono::high_resolution_clock::now();

  int fail_ct = 0;
  Array<State> out_states;
  std::vector<std::mt19937> rand_gens;
  rand_gens.reserve(population);
  for (int i = 0; i < population; i++) {
    rand_gens.push_back(std::mt19937(rand_gen()));
  }

  std::unordered_set<std::string> explored_state_strs;
  size_t iter = 1;
  size_t unchange_cnt = 0;
  while (static_cast<int>(out_states.size()) < sample_init_min_pop_) {
    std::vector<State> temp_states(population);

    // Sample a batch of states randomly
    support::parallel_for(0, population, [this, &temp_states, &sketches, &rand_gens](int index) {
      // Randomly choose a sketch
      State tmp_s = sketches[(rand_gens[index])() % sketches.size()];
      // Apply random annotation rules one by one
      bool valid = true;
      for (const auto& rule : init_rules) {
        if (rule->Apply(this, &tmp_s, &rand_gens[index]) ==
            PopulationGenerationRule::ResultKind::kInvalid) {
          valid = false;
          break;
        }
      }
      if (valid) {
        temp_states[index] = std::move(tmp_s);
      }
    });

    // Filter out the states that were failed to apply initial rules
    Array<State> cand_states;
    for (auto tmp_s : temp_states) {
      if (tmp_s.defined()) {
        cand_states.push_back(std::move(tmp_s));
      } else {
        fail_ct++;
      }
    }

    unchange_cnt++;
    if (!cand_states.empty()) {
      // Run the cost model to make filter out states that failed to extract features.
      // This may happen due to illegal schedules or the schedules that uses too much
      // memory on GPU.
      std::vector<float> pop_scores;
      pop_scores.reserve(cand_states.size());
      cand_states = search_task->compute_dag.InferBound(cand_states);
      PruneInvalidState(search_task, &cand_states);
      program_cost_model->Predict(search_task, cand_states, &pop_scores);

      for (size_t i = 0; i < cand_states.size(); i++) {
        const auto state_str = cand_states[i].ToStr();
        if (pop_scores[i] > -1e10 && explored_state_strs.count(state_str) == 0) {
          explored_state_strs.insert(state_str);
          out_states.push_back(std::move(cand_states[i]));
          unchange_cnt = 0;  // Reset the counter once we found a valid state
        } else {
          fail_ct++;
        }
      }
    }

    if (iter % 5 == 0) {
      double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::high_resolution_clock::now() - tic_begin)
                            .count();
      StdCout(verbose) << "Sample Iter: " << iter << std::fixed << std::setprecision(4)
                       << "\t#Pop: " << out_states.size() << "\t#Target: " << sample_init_min_pop_
                       << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                       << std::setprecision(2) << duration << std::endl;
    }

    if (unchange_cnt == 5) {
      // Reduce the target size to avoid too-long time in this phase if no valid state was found
      // in the past iterations
      if (sample_init_min_pop_ > 1) {
        sample_init_min_pop_ /= 2;
        StdCout(verbose) << "#Target has been reduced to " << sample_init_min_pop_
                         << " due to too many failures or duplications" << std::endl;
      }
      unchange_cnt = 0;
    }
    iter++;
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin)
                        .count();
  StdCout(verbose) << "Sample Initial Population\t#s: " << out_states.size()
                   << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                   << std::setprecision(2) << duration << std::endl;
  return out_states;
}

Array<State> SketchPolicyNode::EvolutionarySearch(const Array<State>& init_population,
                                                  int out_size) {
  Array<State> best_states;
  auto tic_begin = std::chrono::high_resolution_clock::now();

  size_t population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
  double mutation_prob = GetDoubleParam(params, SketchParamKey::EvolutionarySearch::mutation_prob);
  int num_iters = GetIntParam(params, SketchParamKey::EvolutionarySearch::num_iters);

  bool is_cost_model_reasonable = !program_cost_model->IsInstance<RandomModelNode>();
  if (!is_cost_model_reasonable && num_iters > 2) {
    num_iters = 2;
    StdCout(verbose) << "GA iteration number has been adjusted to " << num_iters
                     << " due to random cost model" << std::endl;
  }

  // Two ping pong buffers to avoid copy.
  Array<State> states_buf1{init_population}, states_buf2;
  states_buf1.reserve(population);
  states_buf2.reserve(population);
  Array<State>* pnow = &states_buf1;
  Array<State>* pnext = &states_buf2;

  // A heap to keep the best states during evolution
  using StateHeapItem = std::pair<State, float>;
  auto cmp = [](const StateHeapItem& left, const StateHeapItem& right) {
    return left.second > right.second;
  };
  std::vector<StateHeapItem> heap;
  std::unordered_set<std::string> in_heap(measured_states_set_);
  heap.reserve(out_size);

  // auxiliary global variables
  std::vector<float> pop_scores;
  std::vector<double> pop_selection_probs;
  float max_score = -1e-10f;
  pop_scores.reserve(population);
  pop_selection_probs.reserve(population);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // mutation rules
  int mutation_success_ct, mutation_fail_ct;
  mutation_success_ct = mutation_fail_ct = 0;
  std::vector<float> rule_weights;
  std::vector<double> rule_selection_probs;
  for (const auto& rule : mutation_rules) {
    rule_weights.push_back(rule->weight);
  }
  ComputePrefixSumProb(rule_weights, &rule_selection_probs);

  // Genetic Algorithm
  for (int k = 0; k < num_iters + 1; ++k) {
    // Maintain the heap
    *pnow = search_task->compute_dag.InferBound(*pnow);
    PruneInvalidState(search_task, pnow);
    program_cost_model->Predict(search_task, *pnow, &pop_scores);

    for (size_t i = 0; i < pnow->size(); ++i) {
      const State& state = (*pnow)[i];
      std::string state_str = state.ToStr();

      if (in_heap.count(state_str) == 0) {
        if (static_cast<int>(heap.size()) < out_size) {
          heap.emplace_back((*pnow)[i], pop_scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
          in_heap.insert(state_str);
        } else if (pop_scores[i] > heap.front().second) {
          std::string old_state_str = heap.front().first.ToStr();
          in_heap.erase(old_state_str);
          in_heap.insert(state_str);

          std::pop_heap(heap.begin(), heap.end(), cmp);
          heap.back() = StateHeapItem(state, pop_scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
        }
        if (pop_scores[i] > max_score) {
          max_score = pop_scores[i];
        }
      }
    }

    // Print statistical information
    if (k % 5 == 0 || k == num_iters) {
      StdCout(verbose) << "GA Iter: " << k;
      if (!heap.empty()) {
        StdCout(verbose) << std::fixed << std::setprecision(4) << "\tMax score: " << max_score
                         << std::fixed << std::setprecision(4)
                         << "\tMin score: " << heap.front().second;
      } else {
        StdCout(verbose) << "\tMax score: N/A\tMin score: N/A";
      }
      StdCout(verbose) << "\t#Pop: " << heap.size() << "\t#M+: " << mutation_success_ct / (k + 1)
                       << "\t#M-: " << mutation_fail_ct / (k + 1) << std::endl;
    }
    if (k == num_iters) {
      break;
    }

    // Compute selection probability
    ComputePrefixSumProb(pop_scores, &pop_selection_probs);

    // TODO(merrymercy, comaniac): add crossover.

    // Do mutation
    while (pnext->size() < population) {
      State tmp_s = (*pnow)[RandomChoose(pop_selection_probs, &rand_gen)];

      if (dis(rand_gen) < mutation_prob) {
        const auto& rule = mutation_rules[RandomChoose(rule_selection_probs, &rand_gen)];
        if (rule->Apply(this, &tmp_s, &rand_gen) == PopulationGenerationRule::ResultKind::kValid) {
          pnext->push_back(std::move(tmp_s));
          mutation_success_ct++;
        } else {
          mutation_fail_ct++;
        }
      } else {
        pnext->push_back(std::move(tmp_s));
      }
    }

    std::swap(pnext, pnow);
    pnext->clear();
  }

  // Copy best states in the heap to out_states
  std::sort(heap.begin(), heap.end(), cmp);
  for (auto& item : heap) {
    best_states.push_back(std::move(item.first));
  }

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

/********** PreloadCustomSketchRule **********/
TVM_REGISTER_OBJECT_TYPE(PreloadCustomSketchRuleNode);

PreloadCustomSketchRule::PreloadCustomSketchRule(PackedFunc meet_condition_func,
                                                 PackedFunc apply_func, String rule_name) {
  auto node = make_object<PreloadCustomSketchRuleNode>();
  node->meet_condition_func = std::move(meet_condition_func);
  node->apply_func = std::move(apply_func);
  node->rule_name = std::move(rule_name);
  data_ = std::move(node);
}

void PreloadCustomSketchRuleNode::Callback(SearchPolicyNode* policy) {
  CHECK(policy->IsInstance<SketchPolicyNode>());
  auto sketch_policy = dynamic_cast<SketchPolicyNode*>(policy);
  sketch_policy->sketch_rules.push_back(
      new RuleCustomSketch(meet_condition_func, apply_func, rule_name));
  StdCout(policy->verbose) << "Custom sketch rule \"" << rule_name << "\" added." << std::endl;
}

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicy")
    .set_body_typed([](SearchTask task, CostModel program_cost_model, Map<String, ObjectRef> params,
                       int seed, int verbose,
                       Optional<Array<SearchCallback>> init_search_callbacks) {
      return SketchPolicy(task, program_cost_model, params, seed, verbose, init_search_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicyGenerateSketches")
    .set_body_typed([](SketchPolicy policy) { return policy->GenerateSketches(); });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicySampleInitialPopulation")
    .set_body_typed([](SketchPolicy policy) {
      const Array<State>& sketches = policy->GenerateSketches();

      Array<State> init_population = policy->SampleInitPopulation(sketches);
      return init_population;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicyEvolutionarySearch")
    .set_body_typed([](SketchPolicy policy, Array<State> init_population, int out_size) {
      Array<State> states = policy->EvolutionarySearch(init_population, out_size);
      return states;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.PrintTitle").set_body_typed([](std::string title) {
  PrintTitle(title, 1);
});

TVM_REGISTER_GLOBAL("auto_scheduler.PreloadCustomSketchRule")
    .set_body_typed([](PackedFunc meet_condition_func, PackedFunc apply_func, String rule_name) {
      return PreloadCustomSketchRule(meet_condition_func, apply_func, rule_name);
    });

}  // namespace auto_scheduler
}  // namespace tvm
