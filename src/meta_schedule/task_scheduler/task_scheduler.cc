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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

void TaskSchedulerNode::InitializeTask(int task_id) {
  TuneContext task = this->tasks[task_id];
  TVM_PY_LOG(INFO, this->logging_func)
      << "Initializing Task #" << task_id << ": " << task->task_name;
  TVM_PY_LOG(INFO, task->logging_func)
      << "Initializing Task #" << task_id << ": " << task->task_name;
  CHECK(task->mod.defined()) << "ValueError: Require `context.mod`, but it is not defined";
  CHECK(task->space_generator.defined())
      << "ValueError: Require `context.space_generator`, but it is not defined";
  CHECK(task->search_strategy.defined())
      << "ValueError: Require `context.search_strategy`, but it is not defined";
  TVM_PY_LOG(INFO, task->logging_func) << "\n" << tir::AsTVMScript(task->mod);
  task->Initialize();
  Array<tir::Schedule> design_spaces =
      task->space_generator.value()->GenerateDesignSpace(task->mod.value());
  TVM_PY_LOG(INFO, task->logging_func)
      << "Total " << design_spaces.size() << " design space(s) generated";
  for (int i = 0, n = design_spaces.size(); i < n; ++i) {
    tir::Schedule sch = design_spaces[i];
    tir::Trace trace = sch->trace().value();
    trace = trace->Simplified(true);
    TVM_PY_LOG(INFO, task->logging_func) << "Design space #" << i << ":\n"
                                         << tir::AsTVMScript(sch->mod()) << "\n"
                                         << Concat(trace->AsPython(false), "\n");
  }
  task->search_strategy.value()->PreTuning(design_spaces, database, cost_model);
}

void TaskSchedulerNode::Tune() {
  int n_tasks = this->tasks.size();
  for (int task_id = 0; task_id < n_tasks; ++task_id) {
    InitializeTask(task_id);
  }
  int running_tasks = tasks.size();
  for (int task_id; num_trials_already < max_trials && (task_id = NextTaskId()) != -1;) {
    TVM_PY_LOG(INFO, this->logging_func)
        << "Scheduler picks Task #" << task_id << ": " << tasks[task_id]->task_name;
    TuneContext task = tasks[task_id];
    ICHECK(!task->is_terminated);
    ICHECK(!task->runner_futures.defined());
    if (Optional<Array<MeasureCandidate>> candidates =
            task->search_strategy.value()->GenerateMeasureCandidates()) {
      int num_candidates = candidates.value().size();
      task->_SetMeasureCandidates(candidates.value());
      num_trials_already += num_candidates;
      TVM_PY_LOG(INFO, this->logging_func)
          << "Sending " << num_candidates << " sample(s) to builder";
      task->_SendToBuilder(this->builder);
      TVM_PY_LOG(INFO, this->logging_func)
          << "Sending " << num_candidates << " sample(s) to runner";
      task->_SendToRunner(this->runner);
    } else {
      ICHECK(!task->is_terminated);
      task->is_terminated = true;
      --running_tasks;
      TVM_PY_LOG(INFO, this->logging_func)
          << "Task #" << task_id << " has finished. Remaining task(s): " << running_tasks;
    }
  }
  for (int task_id = 0; task_id < n_tasks; ++task_id) {
    TuneContext task = tasks[task_id];
    if (!task->is_terminated) {
      if (task->runner_futures.defined()) {
        JoinRunningTask(task_id);
      }
      task->is_terminated = true;
      --running_tasks;
      TVM_PY_LOG(INFO, this->logging_func)
          << "Task #" << task_id << " has finished. Remaining task(s): " << running_tasks;
    }
    task->search_strategy.value()->PostTuning();
  }
}

void TaskSchedulerNode::TouchTask(int task_id) {
  TuneContext task = tasks[task_id];
  if (!task->is_terminated && task->runner_futures.defined()) {
    for (const RunnerFuture future : task->runner_futures.value()) {
      if (!future->Done()) {
        return;
      }
    }
    this->JoinRunningTask(task_id);
  }
}

Array<RunnerResult> TaskSchedulerNode::JoinRunningTask(int task_id) {
  TuneContext task = tasks[task_id];
  Array<RunnerResult> results = task->_Join();
  for (const MeasureCallback& callback : this->measure_callbacks) {
    callback->Apply(GetRef<TaskScheduler>(this), task_id, task->measure_candidates.value(),
                    task->builder_results.value(), results);
  }
  task->_ClearMeasureState();
  return results;
}

void PyTaskSchedulerNode::Tune() {
  if (f_tune == nullptr) {
    TaskSchedulerNode::Tune();
  } else {
    f_tune();
  }
}

void PyTaskSchedulerNode::InitializeTask(int task_id) {
  if (f_initialize_task == nullptr) {
    TaskSchedulerNode::InitializeTask(task_id);
  } else {
    f_initialize_task(task_id);
  }
}

void PyTaskSchedulerNode::TouchTask(int task_id) {
  if (f_touch_task == nullptr) {
    return TaskSchedulerNode::TouchTask(task_id);
  } else {
    return f_touch_task(task_id);
  }
}

Array<RunnerResult> PyTaskSchedulerNode::JoinRunningTask(int task_id) {
  if (f_join_running_task == nullptr) {
    return TaskSchedulerNode::JoinRunningTask(task_id);
  } else {
    return f_join_running_task(task_id);
  }
}

int PyTaskSchedulerNode::NextTaskId() {
  ICHECK(f_next_task_id != nullptr) << "PyTaskScheduler's NextTaskId method not implemented!";
  return f_next_task_id();
}

TaskScheduler TaskScheduler::PyTaskScheduler(
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
    PyTaskSchedulerNode::FNextTaskId f_next_task_id) {
  ObjectPtr<PyTaskSchedulerNode> n = make_object<PyTaskSchedulerNode>();
  n->tasks = tasks;
  n->builder = builder;
  n->runner = runner;
  n->database = database;
  n->max_trials = max_trials;
  n->cost_model = cost_model;
  if (measure_callbacks.defined()) {
    n->measure_callbacks = measure_callbacks.value();
  } else {
    n->measure_callbacks = {};
  }
  n->logging_func = logging_func;
  n->num_trials_already = 0;
  n->f_tune = f_tune;
  n->f_initialize_task = f_initialize_task;
  n->f_touch_task = f_touch_task;
  n->f_join_running_task = f_join_running_task;
  n->f_next_task_id = f_next_task_id;
  return TaskScheduler(n);
}

TVM_REGISTER_OBJECT_TYPE(TaskSchedulerNode);
TVM_REGISTER_NODE_TYPE(PyTaskSchedulerNode);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerPyTaskScheduler")
    .set_body_typed(TaskScheduler::PyTaskScheduler);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerTune")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::Tune);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerInitializeTask")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::InitializeTask);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerTouchTask")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::TouchTask);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerJoinRunningTask")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::JoinRunningTask);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerNextTaskId")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::NextTaskId);

}  // namespace meta_schedule
}  // namespace tvm
