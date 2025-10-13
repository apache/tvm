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
#include <tvm/ffi/reflection/registry.h>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The round-robin style task scheduler. */
class RoundRobinNode final : public TaskSchedulerNode {
 public:
  /*! \brief The current task id processed. */
  int task_id = -1;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RoundRobinNode>().def_ro("task_id", &RoundRobinNode::task_id);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.RoundRobin", RoundRobinNode, TaskSchedulerNode);

 protected:
  int NextTaskId() final {
    int n_tasks = this->tasks_.size();
    for (int i = 0; i < n_tasks; ++i) {
      this->TouchTask(i);
    }
    for (int i = 0; i < n_tasks; ++i) {
      task_id = (task_id + 1) % n_tasks;
      TaskRecordNode* task = this->tasks_[task_id].get();
      if (!task->is_terminated) {
        if (task->runner_futures.defined()) {
          JoinRunningTask(task_id);
        }
        return task_id;
      }
    }
    return -1;
  }
};

TaskScheduler TaskScheduler::RoundRobin(ffi::Function logger) {
  ObjectPtr<RoundRobinNode> n = ffi::make_object<RoundRobinNode>();
  n->logger = logger;
  n->task_id = -1;
  return TaskScheduler(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { RoundRobinNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("meta_schedule.TaskSchedulerRoundRobin", TaskScheduler::RoundRobin);
}

}  // namespace meta_schedule
}  // namespace tvm
