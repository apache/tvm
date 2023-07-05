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
#include <tvm/meta_schedule/extracted_task.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/function.h>

#include "../te/operation/create_primfunc.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

ExtractedTask::ExtractedTask(String task_name, IRModule mod, Target target,
                             Array<IRModule> dispatched, int weight) {
  ObjectPtr<ExtractedTaskNode> n = make_object<ExtractedTaskNode>();
  n->task_name = task_name;
  n->mod = mod;
  n->target = target;
  n->dispatched = dispatched;
  n->weight = weight;
  data_ = n;
}

TVM_REGISTER_NODE_TYPE(ExtractedTaskNode);
TVM_REGISTER_GLOBAL("meta_schedule.ExtractedTask")
    .set_body_typed([](String task_name, IRModule mod, Target target, Array<IRModule> dispatched,
                       int weight) -> ExtractedTask {
      return ExtractedTask(task_name, mod, target, dispatched, weight);
    });

}  // namespace meta_schedule
}  // namespace tvm
