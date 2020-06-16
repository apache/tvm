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
 * \file ansor/search_policy/meta_tile_rewrite_policy.h
 * \brief The search policy that searches by program sampling and evolutionary search
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_META_TILE_REWRITE_POLICY_H_
#define TVM_ANSOR_SEARCH_POLICY_META_TILE_REWRITE_POLICY_H_

#include <vector>
#include <string>
#include <utility>
#include <unordered_set>
#include <set>
#include "search_policy.h"
#include "../cost_model/cost_model.h"
#include "../utils.h"


namespace tvm {
namespace ansor {

class SketchGenerationRule;

/*! Multi stage search policy */
class MetaTileRewritePolicyNode: public SearchPolicyNode {
 public:
  CostModel program_cost_model;

  /* this->params is used to store the following arguments
   * int evolutionary_search_population    // The population size for evolutionary search
   * int evolutionary_search_mutation_prob // The probability of mutation for evolutionary search
   * int evolutionary_search_num_iters;    // The number of iterations for evolutionary search
   * double local_mutation_use_measured_ratio;   // The maximum percentage of measured states in the initial
   *                                             // population for evolutionary search
   * double eps_greedy;          // Always allocate this percentage of measurements to random sampled states
   * str cpu_multi_level_tiling_structure // The structure of multi-level tiling for CPU
   * str gpu_multi_level_tiling_structure // The structure of multi-level tiling for GPU
   */
  Map<String, ObjectRef> params;
  std::vector<SketchGenerationRule*> sketch_rules;

  static SearchPolicy make(CostModel program_cost_model,
                           Map<String, ObjectRef> params,
                           int seed);

  // Search and make n_trails measurements
  // Return the best state
  State Search(SearchTask task, int n_trials,
               int early_stopping, int num_measure_per_iter,
               int verbose, ProgramMeasurer measurer,
               Array<SearchCallback> pre_search_callbacks) final;

  // Continue search. This is used by JointTuner
  std::pair<Array<MeasureInput>, Array<MeasureResult> > ContinueSearchOneRound(
      SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) final;

  static constexpr const char *_type_key = "ansor.MetaTileRewritePolicy";
  static const std::vector<int> auto_unroll_configs;

  TVM_DECLARE_FINAL_OBJECT_INFO(MetaTileRewritePolicyNode, SearchPolicyNode);

 protected:
  // Pick states from best states and random states with eps-greedy policy
  void PickStatesWithEpsGreedy(std::vector<MeasureInput>* inputs,
                               const std::vector<State>& best_states,
                               const std::vector<State>& random_states, int remaining_n_trials);

 private:
  // Run one round of the search pipeline
  void SearchOneRound(std::vector<State>* best_states,
                      int num_random_states, std::vector<State>* random_states);

  // Synthesize meta tiling structure without tile size
  void GenerateMetaSketch(std::vector<State>* out_states);

  // Sample init population
  void SampleInitPopulation(const std::vector<State>& meta_structures,
      int out_size, std::vector<State>* out_states);

  // Perform evolutionary search
  void EvolutionarySearch(const std::vector<State>& init_population,
      int num_best_states, std::vector<State>* best_states);

  SplitFactorizationMemo split_memo_;  // Memorize split space for Split
  std::mt19937 rand_gen_;              // Random generator
  int num_measure_per_iter_;   // The number of states to measure per iteration
};
TVM_DEFINE_MUTABLE_OBJECT_REF(MetaTileRewritePolicy, MetaTileRewritePolicyNode);

class PreAddCustomRuleNode : public SearchCallbackNode {
 public:
  // TODO(jcf94): Use tvm::runtime::TypedPackedFunc?
  PackedFunc meet_condition_func;
  PackedFunc apply_func;

  static SearchCallback make(PackedFunc meet_condition_func,
                             PackedFunc apply_func);

  void callback(SearchPolicyNode* policy) final;

  static constexpr const char *_type_key = "ansor.PreAddCustomRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(PreAddCustomRuleNode, SearchCallbackNode);
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_META_TILE_REWRITE_POLICY_H_
