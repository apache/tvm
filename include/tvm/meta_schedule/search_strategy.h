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
#ifndef TVM_META_SCHEDULE_SEARCH_STRATEGY_H_
#define TVM_META_SCHEDULE_SEARCH_STRATEGY_H_

#include <tvm/meta_schedule/arg_info.h>
#include <tvm/meta_schedule/cost_model.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/meta_schedule/measure_candidate.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

// Forward declaration
class TuneContext;
class SearchStrategy;

/*!
 * \brief The search strategy for measure candidates generation.
 * \note The relationship between SearchStrategy and other classes are as follows:
      ┌──────────────────────────────────────────────────────────────┐
   ┌──┴───────────────────────────────────────────────────────────┐  │
┌──┴────────────────── Tune Context ───────────────────────────┐  │  │
│                ┌─────────────────────┐                       │  │  │
│                │                     │   Generate            │  │  │
│                │   Space Generator   ├──────────────┐        │  │  │
│                │                     │              │        │  │  │
│                └─────────────────────┘              ▼        │  │  │
│                                                Design Space  │  │  │
│                ┌─────────────────────┐              │        │  │  │
│      Generate  │                     │   Pretuning  │        │  │  │
│    ┌───────────┤   Search Strategy   │◄─────────────┘        │  │  │
│    │           │                     │                       │  ├──┘
│    │           └─────────────────────┘                       ├──┘
└────┼─────────────────────────────────────────────────────────┘
     │
     │
┌────┼──────────────── Managed By Task Scheduler ─────────────────────┐
│    │                                 ┌───────────┐                  │
│    │                      Send to    │           │  Send to         │
│    ▼                  ┌─────────────►│  Builder  ├──────────┐       │
│ Measure Candidate     │   Builder    │           │  Runner  │       │
│    │                  │              └───────────┘          │       │
│    │     ┌────────────┴────────┐                            │       │
│    │     │                     │     ┌───────────┐          │       │
│    └────►│   Task Scheduler    │     │           │          │       │
│          │                     │     │  Runner   │◄─────────┘       │
│          └─────────────────────┘     │           │                  │
│                   ▲                  └─────┬─────┘                  │
│                   │                        │                        │
│                   └───  Runner Future ◄────┘                        │
└─────────────────────────────────────────────────────────────────────┘
*/
class SearchStrategyNode : public runtime::Object {
 public:
  /*! \brief Virtual destructor */
  virtual ~SearchStrategyNode() = default;

  /*!
   * \brief Initialize the search strategy with tuning context.
   * \param context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& context) = 0;

  /*!
   * \brief Pre-tuning for the search strategy.
   * \param max_trials The maximum number of trials.
   * \param num_trials_per_iter The number of trials per iteration.
   * \param design_spaces The design spaces used during tuning process.
   * \param database The database used during tuning process.
   * \param cost_model The cost model used during tuning process.
   * \note Pre-tuning is supposed to be called before the tuning process and after the
   *  initialization. Because the search strategy is stateful, we can always call pretuning
   *  and reset the search strategy.
   */
  virtual void PreTuning(int max_trials, int num_trials_per_iter,
                         const Array<tir::Schedule>& design_spaces,
                         const Optional<Database>& database,
                         const Optional<CostModel>& cost_model) = 0;

  /*!
   * \brief Post-tuning for the search strategy.
   * \note Post-tuning is supposed to be called after the tuning process and before we reset the
   *  search strategy with another pre-tuning. Post-tuning can be empty.
   */
  virtual void PostTuning() = 0;

  /*!
   * \brief Generate measure candidates from design spaces for measurement.
   * \return The measure candidates generated, nullptr if finished.
   */
  virtual Optional<Array<MeasureCandidate>> GenerateMeasureCandidates() = 0;

  /*!
   * \brief Update the search strategy with measurement results.
   * \param measure_candidates The candidates to be measured.
   * \param results The measurement results from the runner.
   */
  virtual void NotifyRunnerResults(const Array<MeasureCandidate>& measure_candidates,
                                   const Array<RunnerResult>& results) = 0;

  /*!
   * \brief Clone the search strategy.
   * \return The cloned search strategy.
   */
  virtual SearchStrategy Clone() const = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchStrategy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchStrategyNode, Object);
};

/*!
 * \brief Managed reference to SearchStrategyNode.
 * \sa SearchStrategyNode
 */
class SearchStrategy : public runtime::ObjectRef {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief The function type of `PreTuning` method.
   */
  using FPreTuning = runtime::TypedPackedFunc<void(
      int max_trials, int num_trials_per_iter, const Array<tir::Schedule>&,
      const Optional<Database>&, const Optional<CostModel>&)>;
  /*! \brief The function type of `PostTuning` method. */
  using FPostTuning = runtime::TypedPackedFunc<void()>;
  /*!
   * \brief The function type of `GenerateMeasureCandidates` method.
   * \return The measure candidates generated, nullptr if finished.
   */
  using FGenerateMeasureCandidates = runtime::TypedPackedFunc<Optional<Array<MeasureCandidate>>()>;
  /*!
   * \brief The function type of `NotifyRunnerResults` method.
   * \param results The measurement results from the runner.
   */
  using FNotifyRunnerResults =
      runtime::TypedPackedFunc<void(const Array<MeasureCandidate>&, const Array<RunnerResult>&)>;
  /*!
   * \brief The function type of `Clone` method.
   * \return The cloned search strategy.
   */
  using FClone = runtime::TypedPackedFunc<SearchStrategy()>;
  /*!
   * \brief Create a search strategy with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_pre_tuning The packed function of `PreTuning`.
   * \param f_post_tuning The packed function of `PostTuning`.
   * \param f_generate_measure_candidates The packed function of `GenerateMeasureCandidates`.
   * \param f_notify_runner_results The packed function of `NotifyRunnerResults`.
   * \param f_clone The packed function of `Clone`.
   * \return The search strategy created.
   */
  TVM_DLL static SearchStrategy PySearchStrategy(
      FInitializeWithTuneContext f_initialize_with_tune_context,  //
      FPreTuning f_pre_tuning,                                    //
      FPostTuning f_post_tuning,                                  //
      FGenerateMeasureCandidates f_generate_measure_candidates,   //
      FNotifyRunnerResults f_notify_runner_results,               //
      FClone f_clone);

  /*!
   * \brief Constructor of replay trace search strategy.
   * \param max_fail_count The max number of failures during trace replaying.
   */
  TVM_DLL static SearchStrategy ReplayTrace(int max_fail_count);

  /*! \brief Constructor of replay func search strategy. */
  TVM_DLL static SearchStrategy ReplayFunc();

  /*!
   * \brief Constructor of evolutionary search strategy.
   * \param population_size The initial sample population.
   * \param init_measured_ratio The ratio of measures samples in initial population.
   * \param init_min_unmeasured The minimal size of unmeasured population in the initial sampling.
   * \param max_fail_count The max number of failure during initial sampling.
   * \param genetic_num_iters The iterations to run the genetic algorithm.
   * \param genetic_mutate_prob The probability of mutation.
   * \param genetic_max_fail_count The maximum number to try evolving the given trace.
   * \param eps_greedy The ratio to select samples in a greedy fashion via their predicted score.
   */
  TVM_DLL static SearchStrategy EvolutionarySearch(int population_size,         //
                                                   double init_measured_ratio,  //
                                                   int init_min_unmeasured,     //
                                                   int max_fail_count,          //
                                                   int genetic_num_iters,       //
                                                   double genetic_mutate_prob,  //
                                                   int genetic_max_fail_count,  //
                                                   double eps_greedy);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchStrategy, ObjectRef, SearchStrategyNode);
};

/*! \brief The python side customizable class for measure candidate generation */
class PySearchStrategyNode : public SearchStrategyNode {
 public:
  using FInitializeWithTuneContext = SearchStrategy::FInitializeWithTuneContext;
  using FPreTuning = SearchStrategy::FPreTuning;
  using FPostTuning = SearchStrategy::FPostTuning;
  using FGenerateMeasureCandidates = SearchStrategy::FGenerateMeasureCandidates;
  using FNotifyRunnerResults = SearchStrategy::FNotifyRunnerResults;
  using FClone = SearchStrategy::FClone;

  /*! \brief The packed function to the `InitializeWithTuneContext` method. */
  FInitializeWithTuneContext f_initialize_with_tune_context;
  /*! \brief The packed function to the `PreTuning` method. */
  FPreTuning f_pre_tuning;
  /*! \brief The packed function to the `PostTuning` method. */
  FPostTuning f_post_tuning;
  /*! \brief The packed function to the `GenerateMeasureCandidates` method. */
  FGenerateMeasureCandidates f_generate_measure_candidates;
  /*! \brief The packed function to the `NotifyRunnerResults` method. */
  FNotifyRunnerResults f_notify_runner_results;
  /*! \brief The packed function to the `Clone` method. */
  FClone f_clone;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_pre_tuning` is not visited
    // `f_post_tuning` is not visited
    // `f_generate_measure_candidates` is not visited
    // `f_notify_runner_results` is not visited
    // `f_clone` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final;
  void PreTuning(int max_trials, int num_trials_per_iter, const Array<tir::Schedule>& design_spaces,
                 const Optional<Database>& database, const Optional<CostModel>& cost_model) final;
  void PostTuning() final;
  Optional<Array<MeasureCandidate>> GenerateMeasureCandidates() final;
  void NotifyRunnerResults(const Array<MeasureCandidate>& measure_candidates,
                           const Array<RunnerResult>& results);
  SearchStrategy Clone() const final;

  static constexpr const char* _type_key = "meta_schedule.PySearchStrategy";
  TVM_DECLARE_FINAL_OBJECT_INFO(PySearchStrategyNode, SearchStrategyNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SEARCH_STRATEGY_H_
