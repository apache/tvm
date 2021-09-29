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
#include <tvm/meta_schedule/runner.h>
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace meta_schedule {

// Forward declaration
class TuneContext;

/*! \brief The schedule (with input shapes) to be measured. */
class MeasureCandidateNode : public runtime::Object {
 public:
  /*! \brief The schedule for measurement. */
  tir::Schedule sch;
  /*! \brief The argument information, e.g., (shape, dtype) for tensors. */
  Array<ArgInfo> args_info;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("sch", &sch);
    v->Visit("args_info", &args_info);
  }

  static constexpr const char* _type_key = "meta_schedule.MeasureCandidate";
  TVM_DECLARE_FINAL_OBJECT_INFO(MeasureCandidateNode, Object);
};

/*!
 * \brief Managed reference to MeasureCandidateNode.
 * \sa MeasureCandidateNode
 */
class MeasureCandidate : public runtime::ObjectRef {
 public:
  /*!
   * \brief Constructor of MeasureCandidate.
   * \param sch The schedule for measurement.
   * \param args_info The argument information, e.g., (shape, dtype) for tensors.
   */
  TVM_DLL MeasureCandidate(tir::Schedule sch, Array<ArgInfo> args_info);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(MeasureCandidate, ObjectRef, MeasureCandidateNode);
};

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
   * \param tune_context The tuning context for initialization.
   * \note This method is supposed to be called only once before every other method.
   */
  virtual void InitializeWithTuneContext(const TuneContext& tune_context) = 0;

  /*!
   * \brief Pre-tuning for the search strategy.
   * \param design_spaces The design spaces for pre-tuning.
   * \note Pre-tuning is supposed to be called before the tuning process and after the
   *  initialization. Because the search strategy is stateful, we can always call pretuning
   *  and reset the search strategy.
   */
  virtual void PreTuning(const Array<tir::Schedule>& design_spaces) = 0;

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
   * \param results The measurement results from the runner.
   */
  virtual void NotifyRunnerResults(const Array<RunnerResult>& results) = 0;

  static constexpr const char* _type_key = "meta_schedule.SearchStrategy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchStrategyNode, Object);
};

/*! \brief The python side customizable class for measure candidate generation */
class PySearchStrategyNode : public SearchStrategyNode {
 public:
  /*!
   * \brief The function type of `InitializeWithTuneContext` method.
   * \param tune_context The tuning context for initialization.
   */
  using FInitializeWithTuneContext = runtime::TypedPackedFunc<void(const TuneContext&)>;
  /*!
   * \brief The function type of `PreTuning` method.
   * \param design_spaces The design spaces for pre-tuning.
   */
  using FPreTuning = runtime::TypedPackedFunc<void(const Array<tir::Schedule>&)>;
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
  using FNotifyRunnerResults = runtime::TypedPackedFunc<void(const Array<RunnerResult>&)>;

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

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_initialize_with_tune_context` is not visited
    // `f_pre_tuning` is not visited
    // `f_post_tuning` is not visited
    // `f_generate_measure_candidates` is not visited
    // `f_notify_runner_results` is not visited
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    this->f_initialize_with_tune_context(context);
  }

  void PreTuning(const Array<tir::Schedule>& design_spaces) final {
    this->f_pre_tuning(design_spaces);
  }

  void PostTuning() final { this->f_post_tuning(); }

  Optional<Array<MeasureCandidate>> GenerateMeasureCandidates() final {
    return this->f_generate_measure_candidates();
  }

  void NotifyRunnerResults(const Array<RunnerResult>& results) final {
    this->f_notify_runner_results(results);
  }

  static constexpr const char* _type_key = "meta_schedule.PySearchStrategy";
  TVM_DECLARE_FINAL_OBJECT_INFO(PySearchStrategyNode, SearchStrategyNode);
};

/*!
 * \brief Managed reference to SearchStrategyNode.
 * \sa SearchStrategyNode
 */
class SearchStrategy : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a search strategy with customized methods on the python-side.
   * \param f_initialize_with_tune_context The packed function of `InitializeWithTuneContext`.
   * \param f_pre_tuning The packed function of `PreTuning`.
   * \param f_post_tuning The packed function of `PostTuning`.
   * \param f_generate_measure_candidates The packed function of `GenerateMeasureCandidates`.
   * \param f_notify_runner_results The packed function of `NotifyRunnerResults`.
   * \return The search strategy created.
   */
  TVM_DLL static SearchStrategy PySearchStrategy(
      PySearchStrategyNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
      PySearchStrategyNode::FPreTuning f_pre_tuning,                                    //
      PySearchStrategyNode::FPostTuning f_post_tuning,                                  //
      PySearchStrategyNode::FGenerateMeasureCandidates f_generate_measure_candidates,   //
      PySearchStrategyNode::FNotifyRunnerResults f_notify_runner_results);

  /*!
   * \brief Constructor of replay trace search strategy.
   * \param num_trials_per_iter The number of trials per iteration, i.e., the batch size.
   * \param num_trials_total The total number of trials for trace replaying.
   */
  TVM_DLL static SearchStrategy ReplayTrace(int num_trials_per_iter, int num_trials_total);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchStrategy, ObjectRef, SearchStrategyNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SEARCH_STRATEGY_H_
