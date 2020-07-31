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
 * The policy randomly samples programs from the space defined by sketches and use evolutionary
 * search to fine-tune them.
 *
 * Reference:
 * Lianmin, Chengfan, Minmin, Zhao, Cody, et al. "Ansor : Generating High-Performance Tensor
 * Programs for Deep Learning." arXiv preprint arXiv:2006.06762 (2020).
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_SEARCH_POLICY_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_SEARCH_POLICY_H_

#include <tvm/auto_scheduler/cost_model.h>
#include <tvm/auto_scheduler/search_policy.h>

#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

class SketchSearchPolicyNode;

/*! \brief The base class for derivation rules used in the sketch generation. */
class SketchGenerationRule {
 public:
  /*! \brief Result enumeration of the condition function. */
  enum class ConditionKind : int {
    /*! \brief Skip this rule and continue to try the next rules. */
    kSkip = 0,
    /*! \brief Apply this rule and continue to try the next rules. */
    kApply = 1,
    /*! \brief Apply this rule and skip the rest rules. */
    kApplyAndSkipRest = 2
  };

  /*!
   * \brief Condition check function of this rule.
   * \param policy The SketchSearchPolicyNode of this rule, some information may be used during
   * the condition checking.
   * \param state The original state to be checked.
   * \param stage_id The index of the stage to process this condition check.
   * \return The condition check result of this rule.
   */
  virtual ConditionKind MeetCondition(const SketchSearchPolicyNode& policy, const State& state,
                                      int stage_id) = 0;

  /*!
   * \brief Apply function of this rule.
   * \param policy The SketchSearchPolicyNode of this rule, some information may be used during
   * the rule applying.
   * \param state The original state to apply this rule.
   * \param stage_id The index of the next stage to apply this rule.
   * \return The state after applying this rule, and index of the next stage.
   */
  virtual std::vector<std::pair<State, int>> Apply(const SketchSearchPolicyNode& policy,
                                                   const State& state, int stage_id) const = 0;
};

/*! \brief The base class for derivation rules used in the initial population. */
class InitPopulationRule {
 public:
  /*! \brief Result enumeration of the apply function. */
  enum class ResultKind : int { kValid = 0, kInvalid = 1 };

  /*!
   * \brief Apply function of this rule.
   * \param policy The SketchSearchPolicyNode of this rule, some member may get changed during the
   * rule applying. (e.g. random number generator)
   * \param state The state to apply this rule, update inplace.
   * \return The result of this rule, indicate if there's any valid state generated.
   */
  virtual ResultKind Apply(SketchSearchPolicyNode* policy, State* state) const = 0;
};

/*! \brief String keys used in parameter map of SketchSearchPolicy. */
struct SketchParamKey {
  /*! \brief Always allocate this percentage of measurements to random sampled states. */
  static constexpr const char* eps_greedy = "eps_greedy";

  struct EvolutionarySearch {
    /*! \brief The population size for evolutionary search. */
    static constexpr const char* population = "evolutionary_search_population";
    /*! \brief The maximum percentage of measured states in the initial population for evolutionary
     * search. */
    static constexpr const char* use_measured_ratio = "evolutionary_search_use_measured_ratio";
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

/*!
 * \brief The search policy that searches in a hierarchical search space defined by sketches.
 * The policy randomly samples programs from the space defined by sketches
 * and use evolutionary search to  fine-tune them.
 */
class SketchSearchPolicyNode : public SearchPolicyNode {
 public:
  /*! \brief The cost model to estimate the complete schedules. */
  CostModel schedule_cost_model;
  /*! \brief The parameters map for this search process. */
  Map<String, ObjectRef> params;
  /*! \brief The rules to generate sketches. */
  std::vector<SketchGenerationRule*> sketch_rules;
  /*! \brief The rules to generate initial states. */
  std::vector<InitPopulationRule*> init_rules;
  /*! \brief Random generator. */
  std::mt19937 rand_gen;
  /*! \brief Memorize split space for Split. */
  SplitFactorizationMemo split_memo;

  State Search(int num_measure_trials, int early_stopping, int num_measures_per_round,
               ProgramMeasurer measurer) final;

  static constexpr const char* _type_key = "auto_scheduler.SketchSearchPolicy";

  TVM_DECLARE_FINAL_OBJECT_INFO(SketchSearchPolicyNode, SearchPolicyNode);

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
   * \brief Generate sketches.
   * \return The generated sketches(states).
   */
  Array<State> GenerateSketches();

  /*!
   * \brief Sample init population.
   * \param sketches The initial sketches to process population.
   * \param out_size The number of expected output states.
   * \return The generated states after initial population.
   */
  Array<State> SampleInitPopulation(const Array<State>& sketches, int out_size);

  /*!
   * \brief Perform evolutionary search.
   * \param init_populations The states generated from init population.
   * \param out_size The number of expected output states.
   * \return The generated states after evolutionary search.
   */
  Array<State> EvolutionarySearch(const Array<State>& init_populations, int out_size);

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
};

/*!
 * \brief Managed reference to SketchSearchPolicyNode.
 * \sa SketchSearchPolicyNode
 */
class SketchSearchPolicy : public SearchPolicy {
 public:
  /*!
   * \brief The constructor.
   * \param task  The SearchTask for the computation declaration.
   * \param schedule_cost_model The cost model for complete programs.
   * \param params The parameters map for this search process.
   * \param seed The random seed of this search process.
   * \param verbose Verbose level. 0 for silent, 1 to output information during schedule
   * search.
   * \param init_search_callbacks SearchCallback to be called before schedule search.
   */
  SketchSearchPolicy(SearchTask task, CostModel schedule_cost_model, Map<String, ObjectRef> params,
                     int seed, int verbose, Optional<Array<SearchCallback>> init_search_callbacks);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SketchSearchPolicy, SearchPolicy, SketchSearchPolicyNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_SEARCH_POLICY_H_
