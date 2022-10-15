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
#ifndef TVM_META_SCHEDULE_TASK_SCHEDULER_H_
#define TVM_META_SCHEDULE_TASK_SCHEDULER_H_

#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/cost_model.h>
#include <tvm/meta_schedule/measure_callback.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/tune_context.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/support/random_engine.h>

#include <string>
#include <vector>

namespace tvm {
namespace meta_schedule {

class TaskRecordNode : public runtime::Object {
 public:
  /*! \brief The tune context of the task. */
  TuneContext ctx{nullptr};
  /*! \brief The weight of the task */
  double task_weight{1.0};
  /*! \brief The FLOP count of the task */
  double flop{1.0};
  /*! \brief Whether the tuning task has been stopped or finished. */
  bool is_terminated = false;
  /*! \brief Builder errors happens in the task */
  int build_error_count = 0;
  /*! \brief Runner errors happens in the task */
  int run_error_count = 0;
  /*! \brief The latency of each run, in milliseconds. */
  std::vector<double> latency_ms = {};
  /*! \brief The measure candidates. */
  Optional<Array<MeasureCandidate>> measure_candidates = NullOpt;
  /*! \brief The building results. */
  Optional<Array<BuilderResult>> builder_results = NullOpt;
  /*! \brief Packed functions to fetch the runner results asynchronously. */
  Optional<Array<RunnerFuture>> runner_futures = NullOpt;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ctx", &ctx);
    v->Visit("task_weight", &task_weight);
    v->Visit("flop", &flop);
    v->Visit("is_terminated", &is_terminated);
    v->Visit("build_error_count", &build_error_count);
    v->Visit("run_error_count", &run_error_count);
    // `latency_ms` is not visited
    v->Visit("measure_candidates", &measure_candidates);
    v->Visit("builder_results", &builder_results);
    v->Visit("runner_futures", &runner_futures);
  }

  static constexpr const char* _type_key = "meta_schedule.TaskRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(TaskRecordNode, Object);
};

/*!
 * \brief Managed reference to TaskRecordNode.
 * \sa TaskRecordNode
 */
class TaskRecord : public runtime::ObjectRef {
 public:
  /*! \brief Constructor */
  explicit TaskRecord(TuneContext task, double task_weight);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskRecord, ObjectRef, TaskRecordNode);
};

/*!
 * \brief The abstract interface of task schedulers.
 * \note The relationship between SpaceGenerator and other classes are as follows:
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
class TaskSchedulerNode : public runtime::Object {
 public:
  /*! \brief The tuning task's logging function. */
  PackedFunc logger;
  /*! \brief Records for each task */
  Array<TaskRecord> tasks_;
  /*! \brief The list of measure callbacks of the scheduler. */
  Array<MeasureCallback> measure_callbacks_;
  /*! \brief The database used in tuning */
  Optional<Database> database_;
  /*! \brief The cost model used in tuning */
  Optional<CostModel> cost_model_;
  /*! \brief The number of remaining tasks to be tuned. */
  int remaining_tasks_;

  /*! \brief The default destructor. */
  virtual ~TaskSchedulerNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `logger` is not visited
    v->Visit("tasks_", &tasks_);
    v->Visit("measure_callbacks_", &measure_callbacks_);
    v->Visit("database_", &database_);
    v->Visit("cost_model_", &cost_model_);
    v->Visit("remaining_tasks_", &remaining_tasks_);
  }

  /*!
   * \brief Fetch the next task id.
   * \return The next task id.
   */
  virtual int NextTaskId() = 0;
  /*!
   * \brief Wait until the task is finished.
   * \param task_id The task id to be joined.
   * \return The results from the runner.
   */
  virtual Array<RunnerResult> JoinRunningTask(int task_id);
  /*!
   * \brief Jointly tune a given list of tasks.
   * \param tasks The tasks to be tuned
   * \param task_weights The weight of each task
   * \param max_trials_global The maximum number of trials to be performed globally
   * \param max_trials_per_task The maximum number of trials to be performed for each task
   * \param num_trials_per_iter The number of trials to be performed in each iteration
   * \param builder The MetaSchedule builder
   * \param runner The MetaSchedule runner
   * \param measure_callbacks The callbacks to be called after each measurement
   * \param database The database used in tuning
   * \param cost_model The cost model used in tuning
   */
  virtual void Tune(Array<TuneContext> tasks,                  //
                    Array<FloatImm> task_weights,              //
                    int max_trials_global,                     //
                    int max_trials_per_task,                   //
                    int num_trials_per_iter,                   //
                    Builder builder,                           //
                    Runner runner,                             //
                    Array<MeasureCallback> measure_callbacks,  //
                    Optional<Database> database,               //
                    Optional<CostModel> cost_model);
  /*!
   * \brief Terminate a task
   * \param task_id The id of the task to be terminated
   */
  void TerminateTask(int task_id);
  /*!
   * \brief Touch the task and update its status
   * \param task_id The task id to be checked.
   */
  void TouchTask(int task_id);
  /*! \brief Print out a human-readable format of the tuning statistics. */
  void PrintTuningStatistics();

  static constexpr const char* _type_key = "meta_schedule.TaskScheduler";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskSchedulerNode, Object);
};

class TaskScheduler;

/*! \brief The task scheduler with customized methods on the python-side. */
class PyTaskSchedulerNode : public TaskSchedulerNode {
 public:
  /*!
   * \brief The function type of `NextTaskId` method.
   * \return The next task id.
   */
  using FNextTaskId = runtime::TypedPackedFunc<int()>;
  /*!
   * \brief The function type of `JoinRunningTask` method.
   * \param task_id The task id to be joined.
   */
  using FJoinRunningTask = runtime::TypedPackedFunc<Array<RunnerResult>(int)>;
  /*! \brief The function type of `Tune` method. */
  using FTune = runtime::TypedPackedFunc<void(Array<TuneContext> tasks,                  //
                                              Array<FloatImm> task_weights,              //
                                              int max_trials_global,                     //
                                              int max_trials_per_task,                   //
                                              int num_trials_per_iter,                   //
                                              Builder builder,                           //
                                              Runner runner,                             //
                                              Array<MeasureCallback> measure_callbacks,  //
                                              Optional<Database> database,               //
                                              Optional<CostModel> cost_model)>;

  /*! \brief The packed function to the `NextTaskId` function. */
  FNextTaskId f_next_task_id;
  /*! \brief The packed function to the `JoinRunningTask` function. */
  FJoinRunningTask f_join_running_task;
  /*! \brief The packed function to the `Tune` function. */
  FTune f_tune;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TaskSchedulerNode::VisitAttrs(v);
    // `f_next_task_id` is not visited
    // `f_join_running_task` is not visited
    // `f_tune` is not visited
  }

  int NextTaskId() final;
  Array<RunnerResult> JoinRunningTask(int task_id) final;
  void Tune(Array<TuneContext> tasks, Array<FloatImm> task_weights, int max_trials_global,
            int max_trials_per_task, int num_trials_per_iter, Builder builder, Runner runner,
            Array<MeasureCallback> measure_callbacks, Optional<Database> database,
            Optional<CostModel> cost_model) final;

  static constexpr const char* _type_key = "meta_schedule.PyTaskScheduler";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyTaskSchedulerNode, TaskSchedulerNode);
};

/*!
 * \brief Managed reference to TaskSchedulerNode.
 * \sa TaskSchedulerNode
 */
class TaskScheduler : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a task scheduler that fetches tasks in a round-robin fashion.
   * \param logger The tuning task's logging function.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler RoundRobin(PackedFunc logger);
  /*!
   * \brief Create a task scheduler that fetches tasks in a gradient based fashion.
   * \param logger The tuning task's logging function.
   * \param alpha The parameter alpha to control gradient computation.
   * \param window_size The parameter to control backward window size.
   * \param seed The random seed.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler GradientBased(PackedFunc logger, double alpha, int window_size,
                                             support::LinearCongruentialEngine::TRandState seed);
  /*!
   * \brief Create a task scheduler with customized methods on the python-side.
   * \param logger The tuning task's logging function.
   * \param f_next_task_id The packed function of `NextTaskId`.
   * \param f_join_running_task The packed function of `JoinRunningTask`.
   * \param f_tune The packed function of `Tune`.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler PyTaskScheduler(
      PackedFunc logger, PyTaskSchedulerNode::FNextTaskId f_next_task_id,
      PyTaskSchedulerNode::FJoinRunningTask f_join_running_task, PyTaskSchedulerNode::FTune f_tune);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskScheduler, ObjectRef, TaskSchedulerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TASK_SCHEDULER_H_
