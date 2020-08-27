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
 * \file auto_scheduler/search_policy/sketch_policy_rules.h
 * \brief Rules defined to generate the sketches and initial sampled states in SketchPolicy.
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_POLICY_RULES_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_POLICY_RULES_H_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/search_task.h>

#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

class SketchPolicyNode;

/********** Sketch Generation Rule **********/

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
   * \param policy The SketchPolicyNode of this rule, some information may be used during
   * the condition checking.
   * \param state The original state to be checked.
   * \param stage_id The index of the stage to process this condition check.
   * \return The condition check result of this rule.
   */
  virtual ConditionKind MeetCondition(const SketchPolicyNode& policy, const State& state,
                                      int stage_id) const = 0;

  /*!
   * \brief Apply function of this rule.
   * \param policy The SketchPolicyNode of this rule, some information may be used during
   * the rule applying.
   * \param state The original state to apply this rule.
   * \param stage_id The index of the next stage to apply this rule.
   * \return The state after applying this rule, and index of the next stage.
   */
  virtual std::vector<std::pair<State, int>> Apply(const SketchPolicyNode& policy,
                                                   const State& state, int stage_id) const = 0;
};

#define DEFINE_SKETCH_GENERATION_RULE(rule_name)                                                 \
  class rule_name : public SketchGenerationRule {                                                \
   public:                                                                                       \
    ConditionKind MeetCondition(const SketchPolicyNode& policy, const State& state,              \
                                int stage_id) const final;                                       \
    std::vector<std::pair<State, int>> Apply(const SketchPolicyNode& policy, const State& state, \
                                             int stage_id) const final;                          \
  };

/*! \brief The rule that simply skips the current stage. It returns an unchanged state and move to
 * the next stage. */
DEFINE_SKETCH_GENERATION_RULE(RuleSkipStage);

/*! \brief The rule that inlines simple elementwise ops.
 * \note This rule only inlines the strictly inlineable stages. Stages marked as not strictly
 * inlineable will have a chance to try different compute at location in InitPopulation later.
 */
DEFINE_SKETCH_GENERATION_RULE(RuleAlwaysInline);

/*! \brief The rule that performs multi-level tiling. */
DEFINE_SKETCH_GENERATION_RULE(RuleMultiLevelTiling);

/*! \brief The rule that performs multi-level tiling and fuses later consumers. */
DEFINE_SKETCH_GENERATION_RULE(RuleMultiLevelTilingWithFusion);

/*! \brief The rule that adds a cache read stage. Mainly used for GPU cooperative fetching,
 * Currently only support 1 to 1 match cache read. */
DEFINE_SKETCH_GENERATION_RULE(RuleAddCacheRead);

/*! \brief The rule that adds a cache write stage. */
DEFINE_SKETCH_GENERATION_RULE(RuleAddCacheWrite);

/*! \brief The rule that adds rfactor stage. */
DEFINE_SKETCH_GENERATION_RULE(RuleAddRfactor);

/*! \brief The rule that deals with compute ops that perform "fake reduction" with const tensors.
 * This kind of op comes from winograd transformation. */
DEFINE_SKETCH_GENERATION_RULE(RuleSimplifyComputeWithConstTensor);

/*! \brief The rule that use cross thread reduction for GPU. */
DEFINE_SKETCH_GENERATION_RULE(RuleCrossThreadReduction);

/*! \brief Handle special cases in Winograd transformation for GPU. We need to change the compute
 * location of the producers of compute ops that perform "fake reduction" with const tensors. */
DEFINE_SKETCH_GENERATION_RULE(RuleSpecialComputeLocationGPU);

/********** Init Population **********/

/*! \brief The base class for derivation rules used in the initial population. */
class PopulationGenerationRule {
 public:
  /*! \brief Result enumeration of the apply function. */
  enum class ResultKind : int { kValid = 0, kInvalid = 1 };

  /*!
   * \brief Apply function of this rule.
   * \param policy The SketchPolicyNode of this rule, some member may get changed during the
   * rule applying. (e.g. random number generator)
   * \param state The state to apply this rule, update inplace.
   * \return The result of this rule, indicate if there's any valid state generated.
   */
  virtual ResultKind Apply(SketchPolicyNode* policy, State* state) const = 0;
};

#define DEFINE_INIT_POPULATION_RULE(rule_name)                            \
  class rule_name : public PopulationGenerationRule {                     \
   public:                                                                \
    ResultKind Apply(SketchPolicyNode* policy, State* state) const final; \
  };

/*! \brief The rule that fills the incomplete SplitSteps. */
DEFINE_INIT_POPULATION_RULE(InitFillTileSize);

/*! \brief The rule that randomly changes the computation location for some stages, which do not
 * need tiling and are not strictly inlineable(e.g. data padding). */
DEFINE_INIT_POPULATION_RULE(InitChangeComputeLocation);

/*! \brief The rule that annotates parallel for CPU. */
DEFINE_INIT_POPULATION_RULE(InitParallel);

/*! \brief The rule that annotates unroll. */
DEFINE_INIT_POPULATION_RULE(InitUnroll);

/*! \brief The rule that annotates vectorization. */
DEFINE_INIT_POPULATION_RULE(InitVectorization);

/*! \brief The rule that annotates thread binding for GPU. */
DEFINE_INIT_POPULATION_RULE(InitThreadBind);

/********** Mutation **********/

/*! \brief The base class for mutation rules used in the evolutionary search. */
class PopulationMutationRule : public PopulationGenerationRule {
 public:
  /*!
   * \brief Get the priority level of this mutation rule.
   * \return The priority level of this mutation rule. Higher the better.
   */
  virtual int GetLevel(const SearchTask& task) const = 0;
};

// A helper to define mutation rules with a constant rule level.
#define DEFINE_MUTATE_POPULATION_RULE(rule_name, rule_level)                \
  class rule_name : public PopulationMutationRule {                         \
   public:                                                                  \
    ResultKind Apply(SketchPolicyNode* policy, State* state) const final;   \
    int GetLevel(const SearchTask& task) const final { return rule_level; } \
  };

/*! \brief The rule that mutates tile size by randomly dividing a tile size by a factor
    and multipling it to another tile size. */
DEFINE_MUTATE_POPULATION_RULE(MutateTileSize, 100);

/*! \brief The rule that mutates the fusion iterators annotated by parallel. */
DEFINE_MUTATE_POPULATION_RULE(MutateParallel, 50);

/*! \brief The rule that mutates the factor of a randomly selected auto max unroll step. */
class MutateMaxUnrollFactor : public PopulationMutationRule {
 public:
  ResultKind Apply(SketchPolicyNode* policy, State* state) const final;
  int GetLevel(const SearchTask& task) const final { return 10; }

  const std::vector<int> cpu_unroll_cands_ = {0, 16, 64, 512, 1024};
  const std::vector<int> gpu_unroll_cands_ = {0, 16, 64, 512};
};

/*! \brief The rule that randomly changes the computation location for some stages, which do not
 * need tiling and are not strictly inlineable(e.g. data padding). */
class MutateComputeLocation : public PopulationMutationRule {
 public:
  ResultKind Apply(SketchPolicyNode* policy, State* state) const final;
  int GetLevel(const SearchTask& task) const final {
    if (IsGPUTask(task)) {
      return 0;
    }
    return 5;
  }
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_POLICY_RULES_H_
