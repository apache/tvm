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

/*!
 * \brief Send the measure candidates to builder.
 * \param builder The builder to send the candidates to.
 * \param context The tuning context.
 * \param candidates The measure candidates.
 * \return An array of the builder results.
 */
Array<BuilderResult> SendToBuilder(const Builder& builder,  //
                                   const TuneContext& context,
                                   const Array<MeasureCandidate>& candidates) {
  Target target = context->target.value();
  Array<BuilderInput> inputs;
  inputs.reserve(candidates.size());
  for (const MeasureCandidate& candidate : candidates) {
    inputs.push_back(BuilderInput(candidate->sch->mod(), target));
  }
  return builder->Build(inputs);
}

/*!
 * \brief Send the built measure candidates to runner.
 * \param runner The runner to send the candidates to.
 * \param context The tuning context.
 * \param candidates The mesure candidates.
 * \param builder_results The builder results.
 * \return An array of the runner results.
 */
Array<RunnerFuture> SendToRunner(const Runner& runner,  //
                                 const TuneContext& context,
                                 const Array<MeasureCandidate>& candidates,
                                 const Array<BuilderResult>& builder_results) {
  Target target = context->target.value();
  ICHECK_EQ(candidates.size(), builder_results.size());
  int n = candidates.size();
  int n_build_errors = 0;
  Array<RunnerInput> inputs;
  inputs.reserve(n);
  for (int i = 0; i < n; ++i) {
    const MeasureCandidate& candidate = candidates[i];
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.defined()) {
      ++n_build_errors;
      continue;
    }
    inputs.push_back(RunnerInput(/*artifact_path=*/builder_result->artifact_path.value(),
                                 /*device_type=*/target->kind->name,
                                 /*args_info=*/candidate->args_info));
  }
  Array<RunnerFuture> futures = runner->Run(inputs);
  if (n_build_errors == 0) {
    return futures;
  }
  Array<RunnerFuture> results;
  results.reserve(n);
  for (int i = 0, j = 0; i < n; ++i) {
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.defined()) {
      results.push_back(RunnerFuture(
          /*f_done=*/[]() -> bool { return true; },
          /*f_result=*/
          [msg = builder_result->error_msg]() -> RunnerResult {
            return RunnerResult(NullOpt, msg);
          }));
    } else {
      results.push_back(futures[j++]);
    }
  }
  return results;
}

void TaskSchedulerNode::InitializeTask(int task_id) {
  TuneContext task = this->tasks[task_id];
  // Derive the values.
  IRModule mod = task->mod.value();
  SpaceGenerator space = task->space_generator.value();
  SearchStrategy strategy = task->search_strategy.value();
  // Initialize Modules.
  space->InitializeWithTuneContext(task);
  strategy->InitializeWithTuneContext(task);
}

void TaskSchedulerNode::Tune() {
  for (int i = 0; i < static_cast<int>(this->tasks.size()); i++) {
    // Check Optional value validity.
    CHECK(tasks[i]->mod.defined()) << "ValueError: Require `context.mod`, but it is not defined";
    CHECK(tasks[i]->space_generator.defined())
        << "ValueError: Require `context.space_generator`, but it is not defined";
    CHECK(tasks[i]->search_strategy.defined())
        << "ValueError: Require `context.search_strategy`, but it is not defined";

    InitializeTask(i);

    tasks[i]->search_strategy.value()->PreTuning(
        tasks[i]->space_generator.value()->GenerateDesignSpace(tasks[i]->mod.value()));
  }

  int running_tasks = tasks.size();
  while (running_tasks > 0) {
    for (int task_id; (task_id = NextTaskId()) != -1;) {
      TuneContext task = tasks[task_id];
      ICHECK(!task->is_stopped);
      ICHECK(!task->runner_futures.defined());
      SearchStrategy strategy = task->search_strategy.value();
      if ((task->measure_candidates = strategy->GenerateMeasureCandidates()).defined()) {
        Array<BuilderResult> builder_results =
            SendToBuilder(this->builder, task, task->measure_candidates.value());
        task->runner_futures =
            SendToRunner(this->runner, task, task->measure_candidates.value(), builder_results);
      } else {
        SetTaskStopped(task_id);
        --running_tasks;
      }
    }
    int n_tasks = this->tasks.size();
    for (int task_id = 0; task_id < n_tasks; ++task_id)
      if (IsTaskRunning(task_id)) {
        TuneContext task = tasks[task_id];
        this->JoinRunningTask(task_id);
        task->search_strategy.value()->PostTuning();
      }
  }
}

void TaskSchedulerNode::SetTaskStopped(int task_id) {
  TuneContext task = tasks[task_id];
  ICHECK(!task->is_stopped);
  task->is_stopped = true;
}

bool TaskSchedulerNode::IsTaskRunning(int task_id) {
  TuneContext task = tasks[task_id];
  if (task->is_stopped || !task->runner_futures.defined()) {
    return false;
  }
  for (const RunnerFuture future : task->runner_futures.value()) {
    if (!future->Done()) {
      return true;
    }
  }
  this->JoinRunningTask(task_id);
  return false;
}

void TaskSchedulerNode::JoinRunningTask(int task_id) {
  TuneContext task = tasks[task_id];
  ICHECK(task->runner_futures.defined());
  Array<RunnerFuture> futures = task->runner_futures.value();
  int n = futures.size();
  Array<RunnerResult> results;
  results.reserve(n);
  for (const RunnerFuture future : task->runner_futures.value()) {
    results.push_back(future->Result());
  }
  task->search_strategy.value()->NotifyRunnerResults(results);
  task->runner_futures = NullOpt;
  // Add to database
  ICHECK(task->measure_candidates.defined());
  ICHECK(results.size() == task->measure_candidates.value().size());
  int index = 0;
  for (const RunnerResult& result : results) {
    if (!result->error_msg.defined() && result->run_secs.defined()) {
      Optional<tir::Trace> trace = task->measure_candidates.value()[index]->sch->trace();
      ICHECK(trace.defined());
      this->database->CommitTuningRecord(TuningRecord(
          /*trace=*/trace.value(),
          /*run_secs=*/result->run_secs.value(),
          /*workload=*/this->database->CommitWorkload(task->mod.value()),
          /*target=*/task->target.value(),
          /*args_info=*/task->measure_candidates.value()[index]->args_info));
    }
    index++;
  }
}

TaskScheduler TaskScheduler::PyTaskScheduler(
    Array<TuneContext> tasks,                                   //
    Builder builder,                                            //
    Runner runner,                                              //
    Database database,                                          //
    PyTaskSchedulerNode::FTune f_tune,                          //
    PyTaskSchedulerNode::FInitializeTask f_initialize_task,     //
    PyTaskSchedulerNode::FSetTaskStopped f_set_task_stopped,    //
    PyTaskSchedulerNode::FIsTaskRunning f_is_task_running,      //
    PyTaskSchedulerNode::FJoinRunningTask f_join_running_task,  //
    PyTaskSchedulerNode::FNextTaskId f_next_task_id) {
  ObjectPtr<PyTaskSchedulerNode> n = make_object<PyTaskSchedulerNode>();
  n->tasks = tasks;
  n->builder = builder;
  n->runner = runner;
  n->database = database;
  n->f_tune = f_tune;
  n->f_initialize_task = f_initialize_task;
  n->f_set_task_stopped = f_set_task_stopped;
  n->f_is_task_running = f_is_task_running;
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
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerSetTaskStopped")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::SetTaskStopped);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerIsTaskRunning")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::IsTaskRunning);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerJoinRunningTask")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::JoinRunningTask);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerNextTaskId")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::NextTaskId);

}  // namespace meta_schedule
}  // namespace tvm
