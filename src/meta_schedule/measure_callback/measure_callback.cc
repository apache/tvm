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

void PyMeasureCallbackNode::Apply(const TaskScheduler& task_scheduler,                     //
                                  int task_id,                                             //
                                  const ffi::Array<MeasureCandidate>& measure_candidates,  //
                                  const ffi::Array<BuilderResult>& builds,                 //
                                  const ffi::Array<RunnerResult>& results) {
  ICHECK(f_apply != nullptr) << "PyMeasureCallback's Apply method not implemented!";
  auto _ = Profiler::TimedScope("MeasureCallback/" + this->f_as_string());
  return f_apply(task_scheduler, task_id, measure_candidates, builds, results);
}

MeasureCallback MeasureCallback::PyMeasureCallback(PyMeasureCallbackNode::FApply f_apply,  //
                                                   PyMeasureCallbackNode::FAsString f_as_string) {
  ObjectPtr<PyMeasureCallbackNode> n = ffi::make_object<PyMeasureCallbackNode>();
  n->f_apply = std::move(f_apply);
  n->f_as_string = std::move(f_as_string);
  return MeasureCallback(n);
}

ffi::Array<MeasureCallback, void> MeasureCallback::Default() {
  return {
      MeasureCallback::AddToDatabase(),
      MeasureCallback::RemoveBuildArtifact(),
      MeasureCallback::UpdateCostModel(),
  };
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PyMeasureCallbackNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<PyMeasureCallbackNode>();
      ICHECK(self);
      PyMeasureCallbackNode::FAsString f_as_string = (*self).f_as_string;
      ICHECK(f_as_string != nullptr) << "PyMeasureCallback's AsString method not implemented!";
      p->stream << f_as_string();
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  MeasureCallbackNode::RegisterReflection();
  PyMeasureCallbackNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("meta_schedule.MeasureCallbackApply", &MeasureCallbackNode::Apply)
      .def("meta_schedule.MeasureCallbackPyMeasureCallback", MeasureCallback::PyMeasureCallback)
      .def("meta_schedule.MeasureCallbackDefault", MeasureCallback::Default);
}

}  // namespace meta_schedule
}  // namespace tvm
