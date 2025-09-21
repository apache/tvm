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

MeasureCandidate::MeasureCandidate(tir::Schedule sch, ffi::Array<ArgInfo> args_info) {
  ObjectPtr<MeasureCandidateNode> n = ffi::make_object<MeasureCandidateNode>();
  n->sch = sch;
  n->args_info = args_info;
  data_ = std::move(n);
}

void PySearchStrategyNode::InitializeWithTuneContext(const TuneContext& context) {
  ICHECK(f_initialize_with_tune_context != nullptr)
      << "PySearchStrategy's InitializeWithTuneContext method not implemented!";
  f_initialize_with_tune_context(context);
}

void PySearchStrategyNode::PreTuning(int max_trials, int num_trials_per_iter,
                                     const ffi::Array<tir::Schedule>& design_spaces,
                                     const ffi::Optional<Database>& database,
                                     const ffi::Optional<CostModel>& cost_model) {
  ICHECK(f_pre_tuning != nullptr) << "PySearchStrategy's PreTuning method not implemented!";
  f_pre_tuning(max_trials, num_trials_per_iter, design_spaces, database, cost_model);
}

void PySearchStrategyNode::PostTuning() {
  ICHECK(f_post_tuning != nullptr) << "PySearchStrategy's PostTuning method not implemented!";
  f_post_tuning();
}

ffi::Optional<ffi::Array<MeasureCandidate>> PySearchStrategyNode::GenerateMeasureCandidates() {
  ICHECK(f_generate_measure_candidates != nullptr)
      << "PySearchStrategy's GenerateMeasureCandidates method not implemented!";
  return f_generate_measure_candidates();
}

void PySearchStrategyNode::NotifyRunnerResults(
    const ffi::Array<MeasureCandidate>& measure_candidates,
    const ffi::Array<RunnerResult>& results) {
  ICHECK(f_notify_runner_results != nullptr)
      << "PySearchStrategy's NotifyRunnerResults method not implemented!";
  f_notify_runner_results(measure_candidates, results);
}

SearchStrategy PySearchStrategyNode::Clone() const {
  ICHECK(f_clone != nullptr) << "PySearchStrategy's Clone method not implemented!";
  return f_clone();
}

SearchStrategy SearchStrategy::PySearchStrategy(
    PySearchStrategyNode::FInitializeWithTuneContext f_initialize_with_tune_context,  //
    PySearchStrategyNode::FPreTuning f_pre_tuning,                                    //
    PySearchStrategyNode::FPostTuning f_post_tuning,                                  //
    PySearchStrategyNode::FGenerateMeasureCandidates f_generate_measure_candidates,   //
    PySearchStrategyNode::FNotifyRunnerResults f_notify_runner_results,               //
    PySearchStrategyNode::FClone f_clone) {
  ObjectPtr<PySearchStrategyNode> n = ffi::make_object<PySearchStrategyNode>();
  n->f_initialize_with_tune_context = f_initialize_with_tune_context;
  n->f_pre_tuning = f_pre_tuning;
  n->f_post_tuning = f_post_tuning;
  n->f_generate_measure_candidates = f_generate_measure_candidates;
  n->f_notify_runner_results = f_notify_runner_results;
  n->f_clone = f_clone;
  return SearchStrategy(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  MeasureCandidateNode::RegisterReflection();
  PySearchStrategyNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("meta_schedule.MeasureCandidate",
           [](tir::Schedule sch, ffi::Optional<ffi::Array<ArgInfo>> args_info) -> MeasureCandidate {
             return MeasureCandidate(sch, args_info.value_or({}));
           })
      .def("meta_schedule.SearchStrategyPySearchStrategy", SearchStrategy::PySearchStrategy)
      .def_method("meta_schedule.SearchStrategyInitializeWithTuneContext",
                  &SearchStrategyNode::InitializeWithTuneContext)
      .def_method("meta_schedule.SearchStrategyPreTuning", &SearchStrategyNode::PreTuning)
      .def_method("meta_schedule.SearchStrategyPostTuning", &SearchStrategyNode::PostTuning)
      .def_method("meta_schedule.SearchStrategyGenerateMeasureCandidates",
                  &SearchStrategyNode::GenerateMeasureCandidates)
      .def_method("meta_schedule.SearchStrategyNotifyRunnerResults",
                  &SearchStrategyNode::NotifyRunnerResults)
      .def_method("meta_schedule.SearchStrategyClone", &SearchStrategyNode::Clone);
}

}  // namespace meta_schedule
}  // namespace tvm
