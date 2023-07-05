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

MeasureCandidate::MeasureCandidate(tir::Schedule sch, Array<ArgInfo> args_info) {
  ObjectPtr<MeasureCandidateNode> n = make_object<MeasureCandidateNode>();
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
                                     const Array<tir::Schedule>& design_spaces,
                                     const Optional<Database>& database,
                                     const Optional<CostModel>& cost_model) {
  ICHECK(f_pre_tuning != nullptr) << "PySearchStrategy's PreTuning method not implemented!";
  f_pre_tuning(max_trials, num_trials_per_iter, design_spaces, database, cost_model);
}

void PySearchStrategyNode::PostTuning() {
  ICHECK(f_post_tuning != nullptr) << "PySearchStrategy's PostTuning method not implemented!";
  f_post_tuning();
}

Optional<Array<MeasureCandidate>> PySearchStrategyNode::GenerateMeasureCandidates() {
  ICHECK(f_generate_measure_candidates != nullptr)
      << "PySearchStrategy's GenerateMeasureCandidates method not implemented!";
  return f_generate_measure_candidates();
}

void PySearchStrategyNode::NotifyRunnerResults(const Array<MeasureCandidate>& measure_candidates,
                                               const Array<RunnerResult>& results) {
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
  ObjectPtr<PySearchStrategyNode> n = make_object<PySearchStrategyNode>();
  n->f_initialize_with_tune_context = f_initialize_with_tune_context;
  n->f_pre_tuning = f_pre_tuning;
  n->f_post_tuning = f_post_tuning;
  n->f_generate_measure_candidates = f_generate_measure_candidates;
  n->f_notify_runner_results = f_notify_runner_results;
  n->f_clone = f_clone;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(MeasureCandidateNode);
TVM_REGISTER_OBJECT_TYPE(SearchStrategyNode);
TVM_REGISTER_NODE_TYPE(PySearchStrategyNode);

TVM_REGISTER_GLOBAL("meta_schedule.MeasureCandidate")
    .set_body_typed([](tir::Schedule sch, Array<ArgInfo> args_info) -> MeasureCandidate {
      return MeasureCandidate(sch, args_info);
    });
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyPySearchStrategy")
    .set_body_typed(SearchStrategy::PySearchStrategy);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyInitializeWithTuneContext")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::InitializeWithTuneContext);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyPreTuning")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::PreTuning);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyPostTuning")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::PostTuning);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyGenerateMeasureCandidates")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::GenerateMeasureCandidates);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyNotifyRunnerResults")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::NotifyRunnerResults);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyClone")
    .set_body_method<SearchStrategy>(&SearchStrategyNode::Clone);

}  // namespace meta_schedule
}  // namespace tvm
