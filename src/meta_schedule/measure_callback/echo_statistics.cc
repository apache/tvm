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
#include <sstream>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

constexpr const double kMaxTime = 1e10;

std::string GetTaskName(const TuneContext& task, int task_id) {
  std::ostringstream os;
  os << "Task #" << task_id << ": " << task->task_name;
  return os.str();
}

struct TaskInfo {
  std::string name;
  double flop = 0.0;
  int trials = -1;
  int best_round = -1;
  double best_ms = kMaxTime;
  double best_gflops = 0.0;
  int error_count = 0;
  PackedFunc logging_func;

  explicit TaskInfo(const String& name, PackedFunc logging_func)
      : name(name), logging_func(logging_func) {}

  void Update(double run_ms) {
    ++trials;
    if (run_ms < best_ms) {
      best_ms = run_ms;
      best_round = trials;
      best_gflops = flop / run_ms / 1e6;
    }
    TVM_PY_LOG(INFO, logging_func) << "[" << name << "] Trial #" << trials   //
                                   << std::fixed << std::setprecision(4)     //
                                   << ": GFLOPs: " << (flop / run_ms / 1e6)  //
                                   << ". Time: " << run_ms << " ms"          //
                                   << ". Best GFLOPs: " << best_gflops;
  }

  void UpdateError(std::string err, const MeasureCandidate& candidate) {
    static const auto* f_proc = runtime::Registry::Get("meta_schedule._process_error_message");
    ICHECK(f_proc != nullptr);
    err = (*f_proc)(err).operator std::string();
    ++error_count;
    ++trials;
    TVM_PY_LOG(INFO, logging_func)
        << "[" << name << "] Trial #" << trials  //
        << std::fixed << std::setprecision(4)    //
        << ": Error in building: " << err << "\n"
        << tir::AsTVMScript(candidate->sch->mod()) << "\n"
        << Concat(candidate->sch->trace().value()->AsPython(false), "\n");
  }
};

class EchoStatisticsNode : public MeasureCallbackNode {
 public:
  void Apply(const TaskScheduler& task_scheduler, int task_id,
             const Array<MeasureCandidate>& measure_candidates,
             const Array<BuilderResult>& builder_results,
             const Array<RunnerResult>& runner_results) final {
    if (this->task_info.empty()) {
      SetupTaskInfo(task_scheduler->tasks);
    }
    auto _ = Profiler::TimedScope("EchoStatistics");
    ICHECK_EQ(measure_candidates.size(), builder_results.size());
    ICHECK_EQ(measure_candidates.size(), runner_results.size());
    int n = measure_candidates.size();
    TuneContext task = task_scheduler->tasks[task_id];
    TaskInfo& info = this->task_info[task_id];
    std::string task_name = GetTaskName(task, task_id);
    for (int i = 0; i < n; ++i) {
      MeasureCandidate candidate = measure_candidates[i];
      BuilderResult builder_result = builder_results[i];
      RunnerResult runner_result = runner_results[i];
      if (Optional<String> err = builder_result->error_msg) {
        info.UpdateError(err.value(), candidate);
      } else if (Optional<String> err = runner_result->error_msg) {
        info.UpdateError(err.value(), candidate);
      } else {
        ICHECK(runner_result->run_secs.defined());
        info.Update(GetRunMsMedian(runner_result));
      }
    }
  }

  void SetupTaskInfo(const Array<TuneContext>& tasks) {
    task_info.reserve(tasks.size());
    int task_id = 0;
    for (const TuneContext& task : tasks) {
      task_info.push_back(TaskInfo(GetTaskName(task, task_id), task->logging_func));
      TaskInfo& info = task_info.back();
      info.flop = tir::EstimateTIRFlops(task->mod.value());
      ++task_id;
    }
  }

  std::vector<TaskInfo> task_info;

  static constexpr const char* _type_key = "meta_schedule.EchoStatistics";
  TVM_DECLARE_FINAL_OBJECT_INFO(EchoStatisticsNode, MeasureCallbackNode);
};

MeasureCallback MeasureCallback::EchoStatistics() {
  ObjectPtr<EchoStatisticsNode> n = make_object<EchoStatisticsNode>();
  return MeasureCallback(n);
}

TVM_REGISTER_NODE_TYPE(EchoStatisticsNode);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureCallbackEchoStatistics")
    .set_body_typed(MeasureCallback::EchoStatistics);

}  // namespace meta_schedule
}  // namespace tvm
