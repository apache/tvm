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
 * \file auto_scheduler/search_policy/sketch_policy.h
 * \brief This search policy constructs a search space according to the compute declaration.
 * It then randomly samples programs from the search space and uses evolutionary search with a
 * learned cost model to fine tune the sampled programs.
 * The final optimized programs are sent to actual hardware for measurement.
 * The above process is repeated until the auto-scheduler runs out of time budget.
 *
 * Reference:
 * L. Zheng, C. Jia, M. Sun, Z. Wu, C. Yu, et al. "Ansor : Generating High-Performance Tensor
 * Programs for Deep Learning." (OSDI 2020).
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_POLICY_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_POLICY_H_

#include <tvm/auto_scheduler/cost_model.h>
#include <tvm/auto_scheduler/search_policy.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sketch_policy_rules.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

/*! \brief String keys used in parameter map of SketchPolicy. */
struct SketchParamKey {
  /*! \brief Always allocate this percentage of measurements to random sampled states. */
  static constexpr const char* eps_greedy = "eps_greedy";
  /*! \brief Retry several times if SearchOneRound gets no valid state. */
  static constexpr const char* empty_retry_count = "retry_search_one_round_on_empty";

  struct SampleInitPopulation {
    /*! \brief The minimal size of valid population in the initial sampling. */
    static constexpr const char* min_population = "sample_init_min_population";
    /*! \brief The maximum percentage of measured states in the initial sampling. */
    static constexpr const char* use_measured_ratio = "sample_init_use_measured_ratio";
  };

  struct EvolutionarySearch {
    /*! \brief The population size of evolutionary search. */
    static constexpr const char* population = "evolutionary_search_population";
    /*! \brief The number of iterations performed by generic algorithm.*/
    static constexpr const char* num_iters = "evolutionary_search_num_iters";
    /*! \brief The mutation probability.*/
    static constexpr const char* mutation_prob = "evolutionary_search_mutation_prob";
  };

  struct MultiLevelTiling {
    /*! \brief The structure of multi-level tiling for CPU. */
    static constexpr const char* cpu_structure = "cpu_multi_level_tiling_structure";
    /*! \brief The structure of multi-level tiling for GPU. */
    static constexpr const char* gpu_structure = "gpu_multi_level_tiling_structure";
  };

  /*! \brief The max inner most split factor. */
  static constexpr const char* max_innermost_split_factor = "max_innermost_split_factor";
  /*! \brief The max vectorize size. */
  static constexpr const char* max_vectorize_size = "max_vectorize_size";
  /*! \brief Whether disable compute location changing. */
  static constexpr const char* disable_change_compute_location = "disable_change_compute_location";
};

class SketchPolicy;

/*!
 * \brief The search policy that searches in a hierarchical search space defined by sketches.
 * The policy randomly samples programs from the space defined by sketches
 * and use evolutionary search to  fine-tune them.
 */
class SketchPolicyNode : public SearchPolicyNode {
 public:
  /*! \brief The cost model to estimate the complete schedules. */
  CostModel program_cost_model;
  /*! \brief The parameters map for this search policy. */
  Map<String, ObjectRef> params;
  /*! \brief The rules to generate sketches. */
  std::vector<SketchGenerationRule*> sketch_rules;
  /*! \brief The rules to generate initial population. */
  std::vector<PopulationGenerationRule*> init_rules;
  /*! \brief The rules to mutate states in the evolutionary search. */
  std::vector<std::shared_ptr<PopulationMutationRule>> mutation_rules;
  /*! \brief Random generator. */
  std::mt19937 rand_gen;
  /*! \brief Memorize split space for Split. */
  SplitFactorizationMemo split_memo;

  State Search(int num_measure_trials, int early_stopping, int num_measures_per_round,
               ProgramMeasurer measurer) final;

  std::pair<Array<MeasureInput>, Array<MeasureResult>> ContinueSearchOneRound(
      int num_measure, ProgramMeasurer measurer) final;

  /*!
   * \brief Generate sketches.
   * \return The generated sketches(states).
   */
  Array<State> GenerateSketches();

  /*!
   * \brief Sample the init population.
   * \param sketches The initial sketches for the sampled population
   * \return The generated states (the initial population).
   */
  Array<State> SampleInitPopulation(const Array<State>& sketches);

  /*!
   * \brief Perform evolutionary search.
   * \param init_populations The states generated from init population.
   * \param out_size The number of expected output states.
   * \return The generated states after evolutionary search.
   */
  Array<State> EvolutionarySearch(const Array<State>& init_populations, int out_size);

  static constexpr const char* _type_key = "auto_scheduler.SketchPolicy";

  TVM_DECLARE_FINAL_OBJECT_INFO(SketchPolicyNode, SearchPolicyNode);

 private:
  /*!
   * \brief Run one round of the search pipeline.
   * \param num_random_states Number of states that are picked randomly, this is used for
   * eps-greedy policy.
   * \param random_states The picked random states, used as one of the output of this function.
   * \return The best several states generated in this search round.
   */
  Array<State> SearchOneRound(int num_random_states, Array<State>* random_states = nullptr);

  /*!
   * \brief Pick states from best states and random states with eps-greedy policy.
   * \param best_states States picked by cost model.
   * \param random_states States picked randomly.
   * \param remaining_n_trials The remaining number of states need to be generated.
   * \return The generated states to be measured, wrapped in MeasureInput.
   */
  Array<MeasureInput> PickStatesWithEpsGreedy(const Array<State>& best_states,
                                              const Array<State>& random_states,
                                              int remaining_n_trials);

  /*! \brief The number of states to measure per iteration. */
  int num_measure_per_iter_;

  /*! \brief The cached sketches */
  Array<State> sketch_cache_;

  /*! \brief The minimul output population of SampleInitPopulation */
  int sample_init_min_pop_;

  friend class SketchPolicy;
};

/*!
 * \brief Managed reference to SketchPolicyNode.
 * \sa SketchPolicyNode
 */
class SketchPolicy : public SearchPolicy {
 public:
  /*!
   * \brief The constructor.
   * \param task  The SearchTask for the computation declaration.
   * \param program_cost_model The cost model for complete programs.
   * \param params The parameters map for this search process.
   * \param seed The random seed of this search process.
   * \param verbose Verbose level. 0 for silent, 1 to output information during schedule
   * search.
   * \param init_search_callbacks SearchCallback to be called before schedule search.
   */
  SketchPolicy(SearchTask task, CostModel program_cost_model, Map<String, ObjectRef> params,
               int seed, int verbose, Optional<Array<SearchCallback>> init_search_callbacks);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SketchPolicy, SearchPolicy, SketchPolicyNode);
};

/*! \brief Pre-search callback function to load custom rules for sketch generation */
class PreloadCustomSketchRuleNode : public SearchCallbackNode {
 public:
  /*! \brief The condition check function of this rule. */
  PackedFunc meet_condition_func;
  /*! \brief The apply function of this rule. */
  PackedFunc apply_func;
  /*! \brief The name of this rule. */
  String rule_name;

  void Callback(SearchPolicyNode* policy) final;

  static constexpr const char* _type_key = "auto_scheduler.PreloadCustomSketchRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(PreloadCustomSketchRuleNode, SearchCallbackNode);
};

/*!
 * \brief Managed reference to PreloadCustomSketchRuleNode.
 * \sa PreloadCustomSketchRuleNode
 */
class PreloadCustomSketchRule : public SearchCallback {
 public:
  /*!
   * \brief The constructor.
   * \param meet_condition_func The condition check function of this rule.
   * \param apply_func The apply function of this rule.
   * \param rule_name The name of this rule.
   */
  PreloadCustomSketchRule(PackedFunc meet_condition_func, PackedFunc apply_func, String rule_name);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PreloadCustomSketchRule, SearchCallback,
                                        PreloadCustomSketchRuleNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_POLICY_H_
