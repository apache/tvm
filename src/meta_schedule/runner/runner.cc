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

RunnerInput::RunnerInput(String artifact_path, String device_type, Array<ArgInfo> args_info) {
  ObjectPtr<RunnerInputNode> n = make_object<RunnerInputNode>();
  n->artifact_path = artifact_path;
  n->device_type = device_type;
  n->args_info = args_info;
  this->data_ = n;
}

RunnerResult::RunnerResult(Optional<Array<FloatImm>> run_secs, Optional<String> error_msg) {
  ObjectPtr<RunnerResultNode> n = make_object<RunnerResultNode>();
  n->run_secs = run_secs;
  n->error_msg = error_msg;
  this->data_ = n;
}

RunnerFuture::RunnerFuture(RunnerFuture::FDone f_done, RunnerFuture::FResult f_result) {
  ObjectPtr<RunnerFutureNode> n = make_object<RunnerFutureNode>();
  n->f_done = f_done;
  n->f_result = f_result;
  this->data_ = n;
}

Runner Runner::PyRunner(Runner::FRun f_run) {
  ObjectPtr<PyRunnerNode> n = make_object<PyRunnerNode>();
  n->f_run = f_run;
  return Runner(n);
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(RunnerInputNode);
TVM_REGISTER_NODE_TYPE(RunnerResultNode);
TVM_REGISTER_NODE_TYPE(RunnerFutureNode);
TVM_REGISTER_OBJECT_TYPE(RunnerNode);
TVM_REGISTER_NODE_TYPE(PyRunnerNode);
TVM_REGISTER_GLOBAL("meta_schedule.RunnerInput")
    .set_body_typed([](String artifact_path, String device_type,
                       Array<ArgInfo> args_info) -> RunnerInput {
      return RunnerInput(artifact_path, device_type, args_info);
    });
TVM_REGISTER_GLOBAL("meta_schedule.RunnerResult")
    .set_body_typed([](Array<FloatImm> run_secs, Optional<String> error_msg) -> RunnerResult {
      return RunnerResult(run_secs, error_msg);
    });
TVM_REGISTER_GLOBAL("meta_schedule.RunnerFuture")
    .set_body_typed([](RunnerFuture::FDone f_done, RunnerFuture::FResult f_result) -> RunnerFuture {
      return RunnerFuture(f_done, f_result);
    });
TVM_REGISTER_GLOBAL("meta_schedule.RunnerFutureDone")
    .set_body_method<RunnerFuture>(&RunnerFutureNode::Done);
TVM_REGISTER_GLOBAL("meta_schedule.RunnerFutureResult")
    .set_body_method<RunnerFuture>(&RunnerFutureNode::Result);
TVM_REGISTER_GLOBAL("meta_schedule.RunnerRun").set_body_method<Runner>(&RunnerNode::Run);
TVM_REGISTER_GLOBAL("meta_schedule.RunnerPyRunner").set_body_typed(Runner::PyRunner);

}  // namespace meta_schedule
}  // namespace tvm
