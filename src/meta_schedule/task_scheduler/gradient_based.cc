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

struct TaskRecord {
  TuneContext task;
  double weight;
  double flop;
  std::vector<double> best_time_cost_history;  // in ms
  int trials;
};

/*! \brief The gradient based task scheduler. */
class GradientBasedNode final : public TaskSchedulerNode {
 public:
  // Parameters used in gradient computation
  double alpha;
  int window_size;

  std::vector<TaskRecord> task_records_;
  std::vector<double> best_time_cost_per_task_;  // in ms
  int num_rounds_already_;
  support::LinearCongruentialEngine::TRandState rand_state_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TaskSchedulerNode::VisitAttrs(v);
    v->Visit("alpha", &alpha);
    v->Visit("window_size", &window_size);
    // `task_records_` is not visited.
    // `best_time_cost_per_task_` is not visited.
    // `num_rounds_already_` is not visited.
    // `rand_state_` is not visited.
  }

  static constexpr const char* _type_key = "meta_schedule.GradientBased";
  TVM_DECLARE_FINAL_OBJECT_INFO(GradientBasedNode, TaskSchedulerNode);

 public:
  std::string TuningStatistics() const {
    std::ostringstream os;
    int n_tasks = task_records_.size();
    int total_trials = 0;
    double total_latency = 0.0;
    support::TablePrinter p;

    if (using_ipython()) {
      p.Row() << "ID"
              << "Name"
              << "FLOP"
              << "Weight"
              << "GFLOPS"
              << "Latency (us)"
              << "Wtd. Latency"
              << "Trials"
              << "Terminated";
    } else {
      p.Row() << "ID"
              << "Name"
              << "FLOP"
              << "Weight"
              << "Speed (GFLOPS)"
              << "Latency (us)"
              << "Weighted Latency (us)"
              << "Trials"
              << "Terminated";
    }

    p.Separator();

    for (int i = 0; i < n_tasks; ++i) {
      const TaskRecord& record = task_records_[i];
      auto row = p.Row();
      int trials = record.trials;
      String task_name = record.task->task_name.value();
      if (using_ipython() && task_name.length() > 23) {
        std::string temp = task_name.c_str();
        temp = temp.substr(0, 20) + "...";
        task_name = String(temp);
      }
      row << /*id=*/i                                     //
          << /*name=*/task_name                           //
          << /*flops=*/static_cast<int64_t>(record.flop)  //
          << /*weight=*/static_cast<int>(record.weight);
      double latency = 1e9;
      if (trials > 0) {
        latency = record.best_time_cost_history.back();
      }
      if (latency >= 1e9) {
        row << /*speed=*/"N/A" << /*latency=*/"N/A" << /*weighted_latency=*/"N/A";
      } else {
        latency *= 1000.0;
        double speed = record.flop / latency / 1000.0;
        double weighted_latency = latency * record.weight;
        row << /*speed=*/speed << /*latency=*/latency << /*weighted_latency=*/weighted_latency;
        total_latency += weighted_latency;
        total_trials += trials;
      }
      row << trials;
      if (tasks[i]->is_terminated) {
        row << "Y";
      } else {
        row << "";
      }
    }
    p.Separator();
    os << p.AsStr()                                                    //
       << "\nProgress: " << total_trials / (max_trials * 0.01) << "%"  //
       << "\nTotal Trials: " << total_trials << " / " << max_trials    //
       << "\nTotal latency (us): " << total_latency                    //
       << "\n";
    return os.str();
  }

  int NextTaskId() final {
    int n_tasks = task_records_.size();
    // Round robin
    if (num_rounds_already_ == 0) {
      TVM_PY_LOG_CLEAR_SCREEN(this->logging_func);
      TVM_PY_LOG(INFO, this->logging_func) << "\n" << this->TuningStatistics();
    }
    if (num_rounds_already_ < n_tasks) {
      return num_rounds_already_++;
    }
    if (num_rounds_already_ == n_tasks) {
      for (int i = 0; i < n_tasks; ++i) {
        this->JoinRunningTask(i);
      }
    }
    ++num_rounds_already_;
    // Check running tasks
    std::vector<int> tasks_alive;
    tasks_alive.reserve(n_tasks);
    for (int i = 0; i < n_tasks; ++i) {
      this->TouchTask(i);
      if (!tasks[i]->is_terminated) {
        tasks_alive.push_back(i);
      }
    }
    if (tasks_alive.empty()) {
      return -1;
    }
    std::vector<double> grad;
    grad.reserve(n_tasks);
    for (int task_id : tasks_alive) {
      const TaskRecord& record = task_records_[task_id];
      const int w = this->window_size;
      int n = record.best_time_cost_history.size();
      ICHECK_GE(n, 1);
      double best = record.best_time_cost_history[n - 1];
      if (best < 1e9) {
        double g1 = (n >= 1 + w) ? (record.best_time_cost_history[n - 1 - w] - best) / w : 0.0;
        double g2 = best / n;
        double g = alpha * g1 + (1 - alpha) * g2;
        grad.push_back(g * record.weight);
      } else {
        // If the best time cost is unavailable, it means some task is not valid. Skip it.
        grad.push_back(-1e9);
      }
    }
    auto max_grad = std::max_element(grad.begin(), grad.end());
    auto min_grad = std::min_element(grad.begin(), grad.end());
    int task_id = -1;
    if (*max_grad == *min_grad) {
      task_id = tasks_alive[tir::SampleInt(&rand_state_, 0, tasks_alive.size())];
    } else {
      task_id = tasks_alive[std::distance(grad.begin(), max_grad)];
    }
    if (tasks[task_id]->runner_futures.defined()) {
      JoinRunningTask(task_id);
    }
    return task_id;
  }

  Array<RunnerResult> JoinRunningTask(int task_id) final {
    TaskRecord& record = task_records_[task_id];
    Array<RunnerResult> results = TaskSchedulerNode::JoinRunningTask(task_id);
    double& best_time_cost = this->best_time_cost_per_task_[task_id];
    for (const RunnerResult& result : results) {
      if (!result->error_msg.defined()) {
        best_time_cost = std::min(best_time_cost, GetRunMsMedian(result));
      }
    }
    record.best_time_cost_history.push_back(best_time_cost);
    record.trials += results.size();
    TVM_PY_LOG_CLEAR_SCREEN(this->logging_func);
    TVM_PY_LOG(INFO, this->logging_func)
        << "[Updated] Task #" << task_id << ": " << record.task->task_name << "\n"
        << this->TuningStatistics();
    return results;
  }
};

TaskScheduler TaskScheduler::GradientBased(Array<TuneContext> tasks,                            //
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
                                           support::LinearCongruentialEngine::TRandState seed) {
  CHECK_EQ(tasks.size(), task_weights.size())
      << "The size of `tasks` should have the same as `task_weights`.";
  int n_tasks = tasks.size();
  std::vector<TaskRecord> task_records;
  task_records.reserve(n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    task_records.push_back(TaskRecord{
        /*task=*/tasks[i],
        /*weights=*/task_weights[i]->value,
        /*flop=*/std::max(1.0, tir::EstimateTIRFlops(tasks[i]->mod.value())),
        /*best_time_cost_history=*/{},
        /*trials=*/0,
    });
  }
  ObjectPtr<GradientBasedNode> n = make_object<GradientBasedNode>();
  n->tasks = tasks;
  n->builder = builder;
  n->runner = runner;
  n->database = database;
  n->max_trials = max_trials;
  n->cost_model = cost_model;
  n->measure_callbacks = measure_callbacks.value_or({});
  n->logging_func = logging_func;
  n->num_trials_already = 0;
  n->alpha = alpha;
  n->window_size = window_size;
  n->task_records_ = std::move(task_records);
  n->best_time_cost_per_task_ = std::vector<double>(n_tasks, 1e100);
  n->num_rounds_already_ = 0;
  support::LinearCongruentialEngine(&n->rand_state_).Seed(seed);
  return TaskScheduler(n);
}

TVM_REGISTER_NODE_TYPE(GradientBasedNode);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerGradientBased")
    .set_body_typed(TaskScheduler::GradientBased);

}  // namespace meta_schedule
}  // namespace tvm
