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
#include <tvm/meta_schedule/database.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/tune_context.h>

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
  Database database{nullptr};

  /*! \brief The default desctructor. */
  virtual ~TaskSchedulerNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tasks", &tasks);
    v->Visit("builder", &builder);
    v->Visit("runner", &runner);
    v->Visit("database", &database);
  }

  /*! \brief Auto-tuning. */
  virtual void Tune();

  /*!
   * \brief Set specific task to be stopped.
   * \param task_id The task id to be stopped.
   */
  virtual void SetTaskStopped(int task_id);

  /*!
   * \brief Check whether the task is running.
   * \param task_id The task id to be checked.
   * \return Whether the task is running.
   */
  virtual bool IsTaskRunning(int task_id);

  /*!
   * \brief Wait until the task is finished.
   * \param task_id The task id to be joined.
   */
  virtual void JoinRunningTask(int task_id);

  /*!
   * \brief Fetch the next task id.
   * \return The next task id.
   */
  virtual int NextTaskId() = 0;

  static constexpr const char* _type_key = "meta_schedule.TaskScheduler";
  TVM_DECLARE_BASE_OBJECT_INFO(TaskSchedulerNode, Object);
};

/*! \brief The task scheduler with customized methods on the python-side. */
class PyTaskSchedulerNode : public TaskSchedulerNode {
 public:
  /*! \brief The function type of `Tune` method. */
  using FTune = runtime::TypedPackedFunc<void()>;

  /*!
   * \brief The function type of `SetTaskStopped` method.
   * \param task_id The task id to be stopped.
   */
  using FSetTaskStopped = runtime::TypedPackedFunc<void(int)>;

  /*!
   * \brief The function type of `IsTaskRunning` method.
   * \param task_id The task id to be checked.
   * \return Whether the task is running.
   */
  using FIsTaskRunning = runtime::TypedPackedFunc<bool(int)>;

  /*!
   * \brief The function type of `JoinRunningTask` method.
   * \param task_id The task id to be joined.
   */
  using FJoinRunningTask = runtime::TypedPackedFunc<void(int)>;

  /*!
   * \brief The function type of `NextTaskId` method.
   * \return The next task id.
   */
  using FNextTaskId = runtime::TypedPackedFunc<int()>;

  /*! \brief The packed function to the `Tune` funcion. */
  FTune f_tune;
  /*! \brief The packed function to the `SetTaskStopped` function. */
  FSetTaskStopped f_set_task_stopped;
  /*! \brief The packed function to the `IsTaskRunning` function. */
  FIsTaskRunning f_is_task_running;
  /*! \brief The packed function to the `JoinRunningTask` function. */
  FJoinRunningTask f_join_running_task;
  /*! \brief The packed function to the `NextTaskId` function. */
  FNextTaskId f_next_task_id;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_tune` is not visited
    // `f_set_task_stopped` is not visited
    // `f_is_task_running` is not visited
    // `f_join_running_task` is not visited
    // `f_next_task_id` is not visited
  }

  void Tune() final {  //
    f_tune();
  }

  void SetTaskStopped(int task_id) final {  //
    f_set_task_stopped(task_id);
  }

  bool IsTaskRunning(int task_id) final {  //
    return f_is_task_running(task_id);
  }

  void JoinRunningTask(int task_id) final {  //
    f_join_running_task(task_id);
  }

  int NextTaskId() final {  //
    return f_next_task_id();
  }

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
   */
  TVM_DLL static TaskScheduler RoundRobin(Array<TuneContext> tasks, Builder builder, Runner runner,
                                          Database database);
  TVM_DLL static TaskScheduler PyTaskScheduler(
      PyTaskSchedulerNode::FTune f_tune,                          //
      PyTaskSchedulerNode::FSetTaskStopped f_set_task_stopped,    //
      PyTaskSchedulerNode::FIsTaskRunning f_is_task_running,      //
      PyTaskSchedulerNode::FJoinRunningTask f_join_running_task,  //
      PyTaskSchedulerNode::FNextTaskId f_next_task_id);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TaskScheduler, ObjectRef, TaskSchedulerNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TASK_SCHEDULER_H_
