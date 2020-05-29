/*!
 *  Copyright (c) 2020 by Contributors
 * \file ansor/meta_tile_rewrite_policy.h
 * \brief A search policy that search with meta tiling structure and random rewrite
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
  Map<std::string, ObjectRef> params;

  static SearchPolicy make(CostModel program_cost_model,
                           Map<std::string, ObjectRef> params,
                           int seed);

  // Search and make n_trails measurements
  // Return the best state
  State Search(SearchTask task, int n_trials,
               int early_stopping, int num_measure_per_iter,
               int verbose, ProgramMeasurer measurer) final;

  // Continue search. This is used by JointTuner
  std::pair<Array<MeasureInput>, Array<MeasureResult> > ContinueSearchOneRound(
      SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) final;

  static constexpr const char *_type_key = "ansor.MetaTileRewritePolicy";
  static const std::vector<int> auto_unroll_configs;

  TVM_DECLARE_FINAL_OBJECT_INFO(MetaTileRewritePolicyNode, SearchPolicyNode);

  SearchTask cur_task_;                // The current task

  friend class MetaTileRewritePolicyNodeTest;   // Hack friend class for UT
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
  void SynthesizeMetaStructure(std::vector<State>* out_states);

  // Sample init population
  void SampleInitPopulation(const std::vector<State>& meta_structures,
      int out_size, std::vector<State>* out_states);

  // Perform evolutionary search
  void EvolutionarySearch(const std::vector<State>& init_population,
      int num_best_states, std::vector<State>* best_states);

  SplitFactorizationMemo split_memo_;  // Memorize split space for Split
  std::mt19937 rand_gen_;              // Random generator
  int verbose_;                        // Verbose level (0 means silent)
  int num_measure_per_iter_;   // The number of states to measure per iteration

  // The set of the already measured states. We store the string format for redundancy check
  std::unordered_set<std::string> measured_states_set_;

  // The array of already measured states.
  std::vector<State> measured_states_vector_;

  // The throughputs of already measured states
  std::vector<float> measured_states_throughputs_;
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_META_TILE_REWRITE_POLICY_H_
