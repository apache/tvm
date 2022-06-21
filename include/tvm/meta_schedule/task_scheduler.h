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
#include <tvm/meta_schedule/database.h>
#include <tvm/meta_schedule/measure_callback.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/tune_context.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/support/random_engine.h>

namespace tvm {
namespace meta_schedule {

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
  /*! \brief The tasks to be tuned */
  Array<TuneContext> tasks;
  /*! \brief The builder of the scheduler. */
  Builder builder{nullptr};
  /*! \brief The runner of the scheduler. */
  Runner runner{nullptr};
  /*! \brief The database of the scheduler. */
  Optional<Database> database;
  /*! \brief The cost model of the scheduler. */
  Optional<CostModel> cost_model;
  /*! \brief The list of measure callbacks of the scheduler. */
  Array<MeasureCallback> measure_callbacks;
  /*! \brief The maximum number of trials allowed. */
  int max_trials;
  /*! \brief The number of trials already conducted. */
  int num_trials_already;
  /*! \brief The tuning task's logging function. t*/
  PackedFunc logging_func;

  /*! \brief The default destructor. */
  virtual ~TaskSchedulerNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tasks", &tasks);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("database", &database);
    v->Visit("cost_model", &cost_model);
    v->Visit("measure_callbacks", &measure_callbacks);
    v->Visit("max_trials", &max_trials);
    v->Visit("num_trials_already", &num_trials_already);
    // `logging_func` is not visited
  }

  /*! \brief Auto-tuning. */
  virtual void Tune();

  /*!
   * \brief Initialize modules of the given task.
   * \param task_id The task id to be initialized.
   */
  virtual void InitializeTask(int task_id);

  /*!
   * \brief Touch the task and update its status
   * \param task_id The task id to be checked.
   */
  virtual void TouchTask(int task_id);

  /*!
   * \brief Wait until the task is finished.
   * \param task_id The task id to be joined.
   */
  virtual Array<RunnerResult> JoinRunningTask(int task_id);

  /*!
   * \brief Fetch the next task id.
   * \return The next task id.
   */
  virtual int NextTaskId() = 0;

  static constexpr const char* _type_key = "meta_schedule.TaskScheduler";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskSchedulerNode, Object);
};

class TaskScheduler;

/*! \brief The task scheduler with customized methods on the python-side. */
class PyTaskSchedulerNode : public TaskSchedulerNode {
 public:
  /*! \brief The function type of `Tune` method. */
  using FTune = runtime::TypedPackedFunc<void()>;

  /*! \brief The function type of `InitializeTask` method. */
  using FInitializeTask = runtime::TypedPackedFunc<void(int)>;

  /*!
   * \brief The function type of `TouchTask` method.
   * \param task_id The task id to be checked.
   * \return Whether the task is running.
   */
  using FTouchTask = runtime::TypedPackedFunc<void(int)>;

  /*!
   * \brief The function type of `JoinRunningTask` method.
   * \param task_id The task id to be joined.
   */
  using FJoinRunningTask = runtime::TypedPackedFunc<Array<RunnerResult>(int)>;

  /*!
   * \brief The function type of `NextTaskId` method.
   * \return The next task id.
   */
  using FNextTaskId = runtime::TypedPackedFunc<int()>;

  /*! \brief The packed function to the `Tune` function. */
  FTune f_tune;
  /*! \brief The packed function to the `InitializeTask` function. */
  FInitializeTask f_initialize_task;
  /*! \brief The packed function to the `TouchTask` function. */
  FTouchTask f_touch_task;
  /*! \brief The packed function to the `JoinRunningTask` function. */
  FJoinRunningTask f_join_running_task;
  /*! \brief The packed function to the `NextTaskId` function. */
  FNextTaskId f_next_task_id;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_tune` is not visited
    // `f_initialize_task` is not visited
    // `f_touch_task` is not visited
    // `f_join_running_task` is not visited
    // `f_next_task_id` is not visited
  }

  void Tune() final;
  void InitializeTask(int task_id) final;
  void TouchTask(int task_id) final;
  Array<RunnerResult> JoinRunningTask(int task_id) final;
  int NextTaskId() final;

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
   * \param tasks The tasks to be tuned.
   * \param builder The builder of the scheduler.
   * \param runner The runner of the scheduler.
   * \param database The database of the scheduler.
   * \param max_trials The maximum number of trials.
   * \param cost_model The cost model of the scheduler.
   * \param measure_callbacks The measure callbacks of the scheduler.
   * \param logging_func The tuning task's logging function.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler RoundRobin(Array<TuneContext> tasks,                            //
                                          Builder builder,                                     //
                                          Runner runner,                                       //
                                          Optional<Database> database,                         //
                                          Optional<CostModel> cost_model,                      //
                                          Optional<Array<MeasureCallback>> measure_callbacks,  //
                                          int max_trials,                                      //
                                          PackedFunc logging_func);
  /*!
   * \brief Create a task scheduler that fetches tasks in a gradient based fashion.
   * \param tasks The tasks to be tuned.
   * \param task_weights The weights of each task.
   * \param builder The builder of the scheduler.
   * \param runner The runner of the scheduler.
   * \param database The database of the scheduler.
   * \param max_trials The maximum number of trials.
   * \param cost_model The cost model of the scheduler.
   * \param measure_callbacks The measure callbacks of the scheduler.
   * \param logging_func The tuning task's logging function.
   * \param alpha The parameter alpha to control gradient computation.
   * \param window_size The parameter to control backward window size.
   * \param seed The random seed.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler GradientBased(Array<TuneContext> tasks,
                                             Array<FloatImm> task_weights,                        //
                                             Builder builder,                                     //
                                             Runner runner,                                       //
                                             Optional<Database> database,                         //
                                             Optional<CostModel> cost_model,                      //
                                             Optional<Array<MeasureCallback>> measure_callbacks,  //
                                             int max_trials,                                      //
                                             PackedFunc logging_func,                             //
                                             double alpha,                                        //
                                             int window_size,                                     //
                                             support::LinearCongruentialEngine::TRandState seed);
  /*!
   * \brief Create a task scheduler with customized methods on the python-side.
   * \param tasks The tasks to be tuned.
   * \param builder The builder of the scheduler.
   * \param runner The runner of the scheduler.
   * \param database The database of the scheduler.
   * \param max_trials The maximum number of trials.
   * \param cost_model The cost model of the scheduler.
   * \param measure_callbacks The measure callbacks of the scheduler.
   * \param logging_func The tuning task's logging function.
   * \param f_tune The packed function of `Tune`.
   * \param f_initialize_task The packed function of `InitializeTask`.
   * \param f_touch_task The packed function of `TouchTask`.
   * \param f_join_running_task The packed function of `JoinRunningTask`.
   * \param f_next_task_id The packed function of `NextTaskId`.
   * \return The task scheduler created.
   */
  TVM_DLL static TaskScheduler PyTaskScheduler(
      Array<TuneContext> tasks,                                   //
      Builder builder,                                            //
      Runner runner,                                              //
      Optional<Database> database,                                //
      Optional<CostModel> cost_model,                             //
      Optional<Array<MeasureCallback>> measure_callbacks,         //
      int max_trials,                                             //
      PackedFunc logging_func,                                    //
      PyTaskSchedulerNode::FTune f_tune,                          //
      PyTaskSchedulerNode::FInitializeTask f_initialize_task,     //
      PyTaskSchedulerNode::FTouchTask f_touch_task,               //
      PyTaskSchedulerNode::FJoinRunningTask f_join_running_task,  //
      PyTaskSchedulerNode::FNextTaskId f_next_task_id);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskScheduler, ObjectRef, TaskSchedulerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TASK_SCHEDULER_H_
