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

class UpdateCostModelNode : public MeasureCallbackNode {
 public:
  void Apply(const TaskScheduler& task_scheduler, int task_id,
             const Array<MeasureCandidate>& measure_candidates,
             const Array<BuilderResult>& builder_results,
             const Array<RunnerResult>& runner_results) final {
    TuneContext task = task_scheduler->tasks[task_id];
    ICHECK(task_scheduler->cost_model.defined())  //
        << "Cost model must be defined for the task scheduler!";
    ICHECK(task->measure_candidates.defined())  //
        << "Task's measure candidates must be present!";
    CostModel cost_model = task_scheduler->cost_model.value();
    cost_model->Update(task, task->measure_candidates.value(), runner_results);
  }

  static constexpr const char* _type_key = "meta_schedule.UpdateCostModel";
  TVM_DECLARE_FINAL_OBJECT_INFO(UpdateCostModelNode, MeasureCallbackNode);
};

MeasureCallback MeasureCallback::UpdateCostModel() {
  ObjectPtr<UpdateCostModelNode> n = make_object<UpdateCostModelNode>();
  return MeasureCallback(n);
}

TVM_REGISTER_NODE_TYPE(UpdateCostModelNode);
TVM_REGISTER_GLOBAL("meta_schedule.MeasureCallbackUpdateCostModel")
    .set_body_typed(MeasureCallback::UpdateCostModel);

}  // namespace meta_schedule
}  // namespace tvm
