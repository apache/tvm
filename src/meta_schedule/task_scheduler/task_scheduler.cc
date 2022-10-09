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

TaskRecord::TaskRecord(TuneContext ctx, double task_weight) {
  ObjectPtr<TaskRecordNode> n = runtime::make_object<TaskRecordNode>();
  n->ctx = ctx;
  n->task_weight = task_weight;
  n->flop = 1.0;
  auto _ = Profiler::TimedScope("InitializeTask");
  CHECK(ctx->mod.defined()) << "ValueError: Require `context.mod`, but it is not defined";
  CHECK(ctx->space_generator.defined())
      << "ValueError: Require `context.space_generator`, but it is not defined";
  CHECK(ctx->search_strategy.defined())
      << "ValueError: Require `context.search_strategy`, but it is not defined";
  TVM_PY_LOG(INFO, ctx->logger) << "\n" << tir::AsTVMScript(ctx->mod);
  ctx->Initialize();
  n->flop = std::max(1.0, tir::EstimateTIRFlops(ctx->mod.value()));
  this->data_ = std::move(n);
}

void SendToBuilder(TaskRecordNode* self, const Builder& builder) {
  auto _ = Profiler::TimedScope("SendToBuilder");
  Array<MeasureCandidate> candidates = self->measure_candidates.value();
  Target target = self->ctx->target.value();
  Array<BuilderInput> inputs;
  inputs.reserve(candidates.size());
  for (const MeasureCandidate& candidate : candidates) {
    inputs.push_back(BuilderInput(candidate->sch->mod(), target));
  }
  self->builder_results = builder->Build(inputs);
}

void SendToRunner(TaskRecordNode* self, const Runner& runner) {
  auto _ = Profiler::TimedScope("SendToRunner");
  Array<MeasureCandidate> candidates = self->measure_candidates.value();
  Array<BuilderResult> builder_results = self->builder_results.value();
  Target target = self->ctx->target.value();
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
    self->runner_futures = futures;
    return;
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
  self->runner_futures = results;
}

void TaskCleanUp(TaskRecordNode* self, int task_id, const Array<RunnerResult>& results) {
  ICHECK_EQ(self->builder_results.value().size(), results.size());
  ICHECK_EQ(self->runner_futures.value().size(), results.size());
  int n = results.size();
  std::string name = self->ctx->task_name.value();
  const PackedFunc& logger = self->ctx->logger;
  for (int i = 0; i < n; ++i) {
    const BuilderResult& builder_result = self->builder_results.value()[i];
    const MeasureCandidate& candidate = self->measure_candidates.value()[i];
    const RunnerResult& runner_result = results[i];
    Optional<String> error_msg = NullOpt;
    int trials = self->latency_ms.size() + 1;
    double run_ms = 1e9;
    if ((error_msg = builder_result->error_msg)) {
      ++self->build_error_count;
    } else if ((error_msg = runner_result->error_msg)) {
      ++self->run_error_count;
    } else {
      run_ms = GetRunMsMedian(runner_result);
    }
    self->latency_ms.push_back(run_ms);
    if (error_msg) {
      const tir::Schedule& sch = candidate->sch;
      std::string err = error_msg.value();
      TVM_PY_LOG(INFO, logger) << std::fixed << std::setprecision(4)  //
                               << "[Task #" << task_id << ": " << name << "] Trial #" << trials
                               << ": Error in building:\n"
                               << err << "\n"
                               << tir::AsTVMScript(sch->mod()) << "\n"
                               << Concat(sch->trace().value()->AsPython(false), "\n");
    } else {
      double best_ms = *std::min_element(self->latency_ms.begin(), self->latency_ms.end());
      TVM_PY_LOG(INFO, logger) << std::fixed << std::setprecision(4)  //
                               << "[Task #" << task_id << ": " << name << "] Trial #" << trials
                               << ": GFLOPs: " << (self->flop / run_ms / 1e6)
                               << ". Time: " << (run_ms * 1e3) << " us"
                               << ". Best GFLOPs: " << (self->flop / best_ms / 1e6);
    }
  }
  self->measure_candidates = NullOpt;
  self->builder_results = NullOpt;
  self->runner_futures = NullOpt;
}

void TaskSchedulerNode::Tune(Array<TuneContext> ctxs, Array<FloatImm> task_weights,
                             int max_trials_global, int max_trials_per_task,
                             int num_trials_per_iter, Builder builder, Runner runner,
                             Array<MeasureCallback> measure_callbacks, Optional<Database> database,
                             Optional<CostModel> cost_model) {
  CHECK_EQ(ctxs.size(), task_weights.size()) << "ValueError: `task_weights` must have the same "
                                                "length as `ctxs`";
  int n_tasks = this->remaining_tasks_ = ctxs.size();
  this->measure_callbacks_ = measure_callbacks;
  this->database_ = database;
  this->cost_model_ = cost_model;
  this->tasks_.clear();
  this->tasks_.reserve(n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    const TuneContext& ctx = ctxs[i];
    double weight = task_weights[i]->value;
    TVM_PY_LOG(INFO, this->logger) << "Initializing Task #" << i << ": " << ctx->task_name;
    TVM_PY_LOG(INFO, ctx->logger) << "Initializing Task #" << i << ": " << ctx->task_name;
    this->tasks_.push_back(TaskRecord(ctx, weight));
    Array<tir::Schedule> design_spaces =
        ctx->space_generator.value()->GenerateDesignSpace(ctx->mod.value());
    TVM_PY_LOG(INFO, ctx->logger) << "Total " << design_spaces.size()
                                  << " design space(s) generated";
    for (int i = 0, n = design_spaces.size(); i < n; ++i) {
      tir::Schedule sch = design_spaces[i];
      tir::Trace trace = sch->trace().value();
      trace = trace->Simplified(true);
      TVM_PY_LOG(INFO, ctx->logger) << "Design space #" << i << ":\n"
                                    << tir::AsTVMScript(sch->mod()) << "\n"
                                    << Concat(trace->AsPython(false), "\n");
    }
    ctx->search_strategy.value()->PreTuning(max_trials_per_task, num_trials_per_iter, design_spaces,
                                            database, cost_model);
  }

  int num_trials_already = 0;
  for (int task_id; num_trials_already < max_trials_global && (task_id = NextTaskId()) != -1;) {
    TVM_PY_LOG(INFO, this->logger)
        << "TaskScheduler picks Task #" << task_id << ": " << tasks_[task_id]->ctx->task_name;
    TaskRecordNode* task = tasks_[task_id].get();
    ICHECK(!task->is_terminated);
    ICHECK(!task->runner_futures.defined());
    if (static_cast<int>(task->latency_ms.size()) >= max_trials_per_task) {
      TerminateTask(task_id);
      continue;
    }
    if (Optional<Array<MeasureCandidate>> candidates = task->measure_candidates =
            task->ctx->search_strategy.value()->GenerateMeasureCandidates()) {
      int num_candidates = candidates.value().size();
      num_trials_already += num_candidates;
      TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to builder";
      SendToBuilder(task, builder);
      TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to runner";
      SendToRunner(task, runner);
    } else {
      TerminateTask(task_id);
    }
  }
  for (int task_id = 0; task_id < n_tasks; ++task_id) {
    TaskRecordNode* task = this->tasks_[task_id].get();
    if (!task->is_terminated) {
      if (task->runner_futures.defined()) {
        JoinRunningTask(task_id);
      }
      TerminateTask(task_id);
    }
    task->ctx->search_strategy.value()->PostTuning();
  }
}

Array<RunnerResult> TaskSchedulerNode::JoinRunningTask(int task_id) {
  TaskRecordNode* task = this->tasks_[task_id].get();
  ICHECK(task->runner_futures.defined());
  Array<RunnerResult> results;
  {
    auto _ = Profiler::TimedScope("JoinRunnerFutures");
    Array<RunnerFuture> futures = task->runner_futures.value();
    results.reserve(futures.size());
    for (RunnerFuture future : futures) {
      results.push_back(future->Result());
    }
  }
  ICHECK(task->measure_candidates.defined());
  task->ctx->search_strategy.value()->NotifyRunnerResults(task->measure_candidates.value(),
                                                          results);
  ICHECK(task->builder_results.defined());
  ICHECK_EQ(results.size(), task->measure_candidates.value().size());
  ICHECK_EQ(results.size(), task->builder_results.value().size());
  for (const MeasureCallback& callback : this->measure_callbacks_) {
    callback->Apply(GetRef<TaskScheduler>(this), task_id, task->measure_candidates.value(),
                    task->builder_results.value(), results);
  }
  TaskCleanUp(task, task_id, results);
  TVM_PY_LOG_CLEAR_SCREEN(this->logger);
  TVM_PY_LOG(INFO, this->logger) << "[Updated] Task #" << task_id << ": " << task->ctx->task_name
                                 << "\n"
                                 << this->TuningStatistics();
  return results;
}

void TaskSchedulerNode::TouchTask(int task_id) {
  TaskRecordNode* task = this->tasks_[task_id].get();
  if (!task->is_terminated && task->runner_futures.defined()) {
    for (const RunnerFuture future : task->runner_futures.value()) {
      if (!future->Done()) {
        return;
      }
    }
    this->JoinRunningTask(task_id);
  }
}

void TaskSchedulerNode::TerminateTask(int task_id) {
  TaskRecordNode* task = this->tasks_[task_id].get();
  ICHECK(!task->is_terminated);
  task->is_terminated = true;
  --this->remaining_tasks_;
  TVM_PY_LOG_CLEAR_SCREEN(this->logger);
  TVM_PY_LOG(INFO, this->logger) << "Task #" << task_id
                                 << " has finished. Remaining task(s): " << this->remaining_tasks_
                                 << "\n"
                                 << this->TuningStatistics();
}

std::string TaskSchedulerNode::TuningStatistics() const {
  std::ostringstream os;
  int n_tasks = this->tasks_.size();
  int total_trials = 0;
  double total_latency = 0.0;
  support::TablePrinter p;
  p.Row() << "ID"
          << "Name"
          << "FLOP"
          << "Weight"
          << "Speed (GFLOPS)"
          << "Latency (us)"
          << "Weighted Latency (us)"
          << "Trials"
          << "Done";
  p.Separator();
  for (int i = 0; i < n_tasks; ++i) {
    const TaskRecordNode* task = this->tasks_[i].get();
    auto row = p.Row();
    int trials = task->latency_ms.size();
    row << /*id=*/i << /*name=*/task->ctx->task_name.value()  //
        << /*flops=*/static_cast<int64_t>(task->flop)
        << /*weight=*/static_cast<int>(task->task_weight);
    double latency_ms = 1e9;
    if (!task->latency_ms.empty()) {
      latency_ms = *std::min_element(task->latency_ms.begin(), task->latency_ms.end());
    }
    if (latency_ms >= 1e9) {
      row << /*speed=*/"N/A" << /*latency=*/"N/A" << /*weighted_latency=*/"N/A";
    } else {
      latency_ms *= 1000.0;
      double speed = task->flop / latency_ms / 1000.0;
      double weighted_latency = latency_ms * task->task_weight;
      row << /*speed=*/speed << /*latency=*/latency_ms << /*weighted_latency=*/weighted_latency;
      total_latency += weighted_latency;
      total_trials += trials;
    }
    row << trials;
    if (task->is_terminated) {
      row << "Y";
    } else {
      row << "";
    }
  }
  p.Separator();
  os << p.AsStr()                                  //
     << "\nTotal trials: " << total_trials         //
     << "\nTotal latency (us): " << total_latency  //
     << "\n";
  return os.str();
}

TaskScheduler TaskScheduler::PyTaskScheduler(
    PackedFunc logger, PyTaskSchedulerNode::FNextTaskId f_next_task_id,
    PyTaskSchedulerNode::FJoinRunningTask f_join_running_task, PyTaskSchedulerNode::FTune f_tune) {
  CHECK(f_next_task_id != nullptr) << "ValueError: next_task_id is not defined";
  ObjectPtr<PyTaskSchedulerNode> n = make_object<PyTaskSchedulerNode>();
  n->logger = logger;
  n->f_next_task_id = f_next_task_id;
  n->f_join_running_task = f_join_running_task;
  n->f_tune = f_tune;
  return TaskScheduler(n);
}

int PyTaskSchedulerNode::NextTaskId() {
  CHECK(f_next_task_id != nullptr) << "PyTaskScheduler's NextTaskId method not implemented!";
  return f_next_task_id();
}

Array<RunnerResult> PyTaskSchedulerNode::JoinRunningTask(int task_id) {
  if (f_join_running_task == nullptr) {
    return TaskSchedulerNode::JoinRunningTask(task_id);
  } else {
    return f_join_running_task(task_id);
  }
}

void PyTaskSchedulerNode::Tune(Array<TuneContext> tasks, Array<FloatImm> task_weights,
                               int max_trials_global, int max_trials_per_task,
                               int num_trials_per_iter, Builder builder, Runner runner,
                               Array<MeasureCallback> measure_callbacks,
                               Optional<Database> database, Optional<CostModel> cost_model) {
  if (f_tune == nullptr) {
    TaskSchedulerNode::Tune(tasks, task_weights, max_trials_global, max_trials_per_task,
                            num_trials_per_iter, builder, runner, measure_callbacks, database,
                            cost_model);
  } else {
    f_tune(tasks, task_weights, max_trials_global, max_trials_per_task, num_trials_per_iter,
           builder, runner, measure_callbacks, database, cost_model);
  }
}

TVM_REGISTER_NODE_TYPE(TaskRecordNode);
TVM_REGISTER_OBJECT_TYPE(TaskSchedulerNode);
TVM_REGISTER_NODE_TYPE(PyTaskSchedulerNode);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerPyTaskScheduler")
    .set_body_typed(TaskScheduler::PyTaskScheduler);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerTune")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::Tune);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerJoinRunningTask")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::JoinRunningTask);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerNextTaskId")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::NextTaskId);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerTerminateTask")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::TerminateTask);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerTouchTask")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::TouchTask);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerTuningStatistics")
    .set_body_method<TaskScheduler>(&TaskSchedulerNode::TuningStatistics);

}  // namespace meta_schedule
}  // namespace tvm
