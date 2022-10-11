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

/*! \brief The gradient based task scheduler. */
class GradientBasedNode final : public TaskSchedulerNode {
 public:
  double alpha;
  int window_size;
  support::LinearCongruentialEngine::TRandState rand_state;

  int round_robin_rounds_;
  std::vector<std::vector<double>> best_latency_history_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TaskSchedulerNode::VisitAttrs(v);
    v->Visit("alpha", &alpha);
    v->Visit("window_size", &window_size);
    // `rand_state` is not visited.
    // `num_rounds_already_` is not visited.
    // `best_latency_history_` is not visited.
  }

  static constexpr const char* _type_key = "meta_schedule.GradientBased";
  TVM_DECLARE_FINAL_OBJECT_INFO(GradientBasedNode, TaskSchedulerNode);

 public:
  void Tune(Array<TuneContext> tasks, Array<FloatImm> task_weights, int max_trials_global,
            int max_trials_per_task, int num_trials_per_iter, Builder builder, Runner runner,
            Array<MeasureCallback> measure_callbacks, Optional<Database> database,
            Optional<CostModel> cost_model) final {
    int n_tasks = tasks.size();
    round_robin_rounds_ = 0;
    best_latency_history_.resize(n_tasks, std::vector<double>());
    TaskSchedulerNode::Tune(tasks, task_weights, max_trials_global, max_trials_per_task,
                            num_trials_per_iter, builder, runner, measure_callbacks, database,
                            cost_model);
  }

  int NextTaskId() final {
    int n_tasks = this->tasks_.size();
    // Step 1. Check if it's in round robin mode.
    if (round_robin_rounds_ == 0) {
      TVM_PY_LOG_CLEAR_SCREEN(this->logger);
      this->PrintTuningStatistics();
    }
    if (round_robin_rounds_ < n_tasks) {
      return round_robin_rounds_++;
    }
    if (round_robin_rounds_ == n_tasks) {
      for (int i = 0; i < n_tasks; ++i) {
        this->JoinRunningTask(i);
      }
      ++round_robin_rounds_;
    }
    // Step 2. Collect the tasks that are not terminated yet
    std::vector<int> tasks_alive;
    {
      tasks_alive.reserve(n_tasks);
      for (int i = 0; i < n_tasks; ++i) {
        this->TouchTask(i);
        if (!this->tasks_[i]->is_terminated) {
          tasks_alive.push_back(i);
        }
      }
      if (tasks_alive.empty()) {
        return -1;
      }
    }
    // Step 3. Calculate the gradient of each task alive
    std::vector<double> grad;
    grad.reserve(n_tasks);
    for (int task_id : tasks_alive) {
      const std::vector<double>& best_latency = this->best_latency_history_.at(task_id);
      int n = best_latency.size();
      ICHECK_GE(n, 1);
      double task_weight = this->tasks_[task_id]->task_weight;
      int w = this->window_size;
      double best = best_latency[n - 1];
      if (best < 1e9) {
        double g1 = (n >= 1 + w) ? (best_latency[n - 1 - w] - best) / w : 0.0;
        double g2 = best / n;
        double g = alpha * g1 + (1 - alpha) * g2;
        grad.push_back(g * task_weight);
      } else {
        // If the best time cost is unavailable, it means some task is not valid. Skip it.
        grad.push_back(-1e9);
      }
    }
    // Step 4. Select the task with the largest gradient
    auto max_grad = std::max_element(grad.begin(), grad.end());
    auto min_grad = std::min_element(grad.begin(), grad.end());
    int task_id = -1;
    if (*max_grad == *min_grad) {
      task_id = tasks_alive[tir::SampleInt(&this->rand_state, 0, tasks_alive.size())];
    } else {
      task_id = tasks_alive[std::distance(grad.begin(), max_grad)];
    }
    if (this->tasks_[task_id]->runner_futures.defined()) {
      JoinRunningTask(task_id);
    }
    return task_id;
  }

  Array<RunnerResult> JoinRunningTask(int task_id) final {
    Array<RunnerResult> results = TaskSchedulerNode::JoinRunningTask(task_id);
    TaskRecordNode* task = this->tasks_[task_id].get();
    this->best_latency_history_.at(task_id).push_back(
        *std::min_element(task->latency_ms.begin(),  //
                          task->latency_ms.end()));
    return results;
  }
};

TaskScheduler TaskScheduler::GradientBased(PackedFunc logger, double alpha, int window_size,
                                           support::LinearCongruentialEngine::TRandState seed) {
  ObjectPtr<GradientBasedNode> n = make_object<GradientBasedNode>();
  n->logger = logger;
  n->alpha = alpha;
  n->window_size = window_size;
  n->rand_state = support::LinearCongruentialEngine::NormalizeSeed(seed);
  return TaskScheduler(n);
}

TVM_REGISTER_NODE_TYPE(GradientBasedNode);
TVM_REGISTER_GLOBAL("meta_schedule.TaskSchedulerGradientBased")
    .set_body_typed(TaskScheduler::GradientBased);

}  // namespace meta_schedule
}  // namespace tvm
