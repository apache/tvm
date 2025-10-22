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

RunnerInput::RunnerInput(ffi::String artifact_path, ffi::String device_type,
                         ffi::Array<ArgInfo> args_info) {
  ObjectPtr<RunnerInputNode> n = ffi::make_object<RunnerInputNode>();
  n->artifact_path = artifact_path;
  n->device_type = device_type;
  n->args_info = args_info;
  this->data_ = n;
}

RunnerResult::RunnerResult(ffi::Optional<ffi::Array<FloatImm>> run_secs,
                           ffi::Optional<ffi::String> error_msg) {
  ObjectPtr<RunnerResultNode> n = ffi::make_object<RunnerResultNode>();
  n->run_secs = run_secs;
  n->error_msg = error_msg;
  this->data_ = n;
}

RunnerFuture::RunnerFuture(RunnerFuture::FDone f_done, RunnerFuture::FResult f_result) {
  ObjectPtr<RunnerFutureNode> n = ffi::make_object<RunnerFutureNode>();
  n->f_done = f_done;
  n->f_result = f_result;
  this->data_ = n;
}

Runner Runner::PyRunner(Runner::FRun f_run) {
  ObjectPtr<PyRunnerNode> n = ffi::make_object<PyRunnerNode>();
  n->f_run = f_run;
  return Runner(n);
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  RunnerNode::RegisterReflection();
  RunnerInputNode::RegisterReflection();
  RunnerResultNode::RegisterReflection();
  RunnerFutureNode::RegisterReflection();
  PyRunnerNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("meta_schedule.RunnerInput",
           [](ffi::String artifact_path, ffi::String device_type, ffi::Array<ArgInfo> args_info)
               -> RunnerInput { return RunnerInput(artifact_path, device_type, args_info); })
      .def("meta_schedule.RunnerResult",
           [](ffi::Optional<ffi::Array<FloatImm>> run_secs, ffi::Optional<ffi::String> error_msg)
               -> RunnerResult { return RunnerResult(run_secs, error_msg); })
      .def("meta_schedule.RunnerFuture",
           [](RunnerFuture::FDone f_done, RunnerFuture::FResult f_result) -> RunnerFuture {
             return RunnerFuture(f_done, f_result);
           })
      .def_method("meta_schedule.RunnerFutureDone", &RunnerFutureNode::Done)
      .def_method("meta_schedule.RunnerFutureResult", &RunnerFutureNode::Result)
      .def_method("meta_schedule.RunnerRun", &RunnerNode::Run)
      .def("meta_schedule.RunnerPyRunner", Runner::PyRunner);
}

}  // namespace meta_schedule
}  // namespace tvm
