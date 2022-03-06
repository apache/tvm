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
Array<BuilderResult> SendToBuilder(const Builder& builder, const TuneContext& context,
                                   const Array<MeasureCandidate>& candidates) {
  LOG(INFO) << "Sending " << candidates.size() << " sample(s) to builder";
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
 * \param candidates The measure candidates.
 * \param builder_results The builder results.
 * \return An array of the runner results.
 */
Array<RunnerFuture> SendToRunner(const Runner& runner, const TuneContext& context,
                                 const Array<MeasureCandidate>& candidates,
                                 const Array<BuilderResult>& builder_results) {
  LOG(INFO) << "Sending " << candidates.size() << " sample(s) to runner";
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
  LOG(INFO) << "Initializing Task #" << task_id << ": " << task->task_name << ", mod =\n"
            << tir::AsTVMScript(task->mod);
  this->tasks[task_id]->Initialize();
}

void TaskSchedulerNode::Tune() {
  for (int i = 0; i < static_cast<int>(this->tasks.size()); i++) {
    TuneContext task = tasks[i];
    // Check Optional value validity.
    CHECK(task->mod.defined()) << "ValueError: Require `context.mod`, but it is not defined";
    CHECK(task->space_generator.defined())
        << "ValueError: Require `context.space_generator`, but it is not defined";
    CHECK(task->search_strategy.defined())
        << "ValueError: Require `context.search_strategy`, but it is not defined";
    InitializeTask(i);
    Array<tir::Schedule> design_spaces =
        task->space_generator.value()->GenerateDesignSpace(task->mod.value());
    LOG(INFO) << "Total " << design_spaces.size() << " design space(s) generated";
    for (int i = 0, n = design_spaces.size(); i < n; ++i) {
      tir::Schedule sch = design_spaces[i];
      tir::Trace trace = sch->trace().value();
      trace = trace->Simplified(true);
      LOG(INFO) << "Design space #" << i << ":\n"
                << tir::AsTVMScript(sch->mod()) << "\n"
                << Concat(trace->AsPython(false), "\n");
    }
    task->search_strategy.value()->PreTuning(design_spaces);
  }

  int running_tasks = tasks.size();
  for (int task_id; (task_id = NextTaskId()) != -1;) {
    LOG(INFO) << "Scheduler picks Task #" << task_id << ": " << tasks[task_id]->task_name;
    TuneContext task = tasks[task_id];
    ICHECK(!task->is_stopped);
    ICHECK(!task->runner_futures.defined());
    SearchStrategy strategy = task->search_strategy.value();
    if ((task->measure_candidates = strategy->GenerateMeasureCandidates()).defined()) {
      Array<BuilderResult> builder_results =
          SendToBuilder(this->builder, task, task->measure_candidates.value());
      task->builder_results = builder_results;
      task->runner_futures =
          SendToRunner(this->runner, task, task->measure_candidates.value(), builder_results);
    } else {
      SetTaskStopped(task_id);
      --running_tasks;
      LOG(INFO) << "Task #" << task_id << " has finished. Remaining task(s): " << running_tasks;
    }
  }
  ICHECK_EQ(running_tasks, 0) << "Not all tasks are finished";
  int n_tasks = this->tasks.size();
  for (int task_id = 0; task_id < n_tasks; ++task_id) {
    ICHECK(!IsTaskRunning(task_id)) << "Task #" << task_id << " is still running";
    TuneContext task = tasks[task_id];
    task->search_strategy.value()->PostTuning();
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
  task->search_strategy.value()->NotifyRunnerResults(task, task->measure_candidates.value(),
                                                     results);
  // Invoke the callbacks
  ICHECK(task->measure_candidates.defined());
  ICHECK(task->builder_results.defined());
  ICHECK_EQ(results.size(), task->measure_candidates.value().size());
  ICHECK_EQ(results.size(), task->builder_results.value().size());
  for (const MeasureCallback& callback : this->measure_callbacks) {
    callback->Apply(GetRef<TaskScheduler>(this), task_id, task->measure_candidates.value(),
                    task->builder_results.value(), results);
  }
  task->measure_candidates = NullOpt;
  task->builder_results = NullOpt;
  task->runner_futures = NullOpt;
}

TaskScheduler TaskScheduler::PyTaskScheduler(
    Array<TuneContext> tasks,                                   //
    Builder builder,                                            //
    Runner runner,                                              //
    Database database,                                          //
    Optional<CostModel> cost_model,                             //
    Optional<Array<MeasureCallback>> measure_callbacks,         //
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
  n->cost_model = cost_model;
  if (measure_callbacks.defined()) {
    n->measure_callbacks = measure_callbacks.value();
  } else {
    n->measure_callbacks = {};
  }
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
