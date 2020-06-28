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

/*!
 * \file ansor/auto_schedule.cc
 * \brief The user interface of the Ansor auto-scheduler.
 */

#include "auto_schedule.h"
#include <tvm/runtime/registry.h>
#include <string>
#include <utility>

namespace tvm {
namespace ansor {

TVM_REGISTER_NODE_TYPE(TuneOptionNode);

TuneOption::TuneOption(int n_trials, int early_stopping,
                       int num_measure_per_round, int verbose, Builder builder,
                       Runner runner, Array<MeasureCallback> measure_callbacks,
                       Array<SearchCallback> pre_search_callbacks) {
  auto node = make_object<TuneOptionNode>();
  node->n_trials = n_trials;
  node->early_stopping = early_stopping;
  node->num_measure_per_round = num_measure_per_round;
  node->verbose = verbose;
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->measure_callbacks = std::move(measure_callbacks);
  node->pre_search_callbacks = std::move(pre_search_callbacks);
  data_ = std::move(node);
}

std::pair<te::Schedule, Array<te::Tensor> > AutoSchedule(SearchTask task,
    SearchPolicy search_policy, TuneOption tune_option) {
  // Create a ProgramMeasurer to handle the schedule build and performance measure
  ProgramMeasurer measurer =
      ProgramMeasurer(tune_option->builder, tune_option->runner,
                      tune_option->measure_callbacks,
                      tune_option->verbose);
  // Search for the best schedule
  State state = search_policy->Search(
      task, tune_option->n_trials, tune_option->early_stopping,
      tune_option->num_measure_per_round, tune_option->verbose, measurer,
      tune_option->pre_search_callbacks);
  return task->compute_dag.ApplySteps(state->transform_steps);
}

std::pair<te::Schedule, Array<te::Tensor> > AutoSchedule(
    std::string workload_key, Target target, Target target_host,
    SearchPolicy search_policy, HardwareParams hardware_params,
    TuneOption tune_option) {
  // Create SearchTask from the given workload key
  ComputeDAG dag = ComputeDAG(workload_key);
  SearchTask task = SearchTask(
      std::move(dag), std::move(workload_key), std::move(target),
      std::move(target_host), std::move(hardware_params));
  // Search for the best schedule
  return AutoSchedule(std::move(task), std::move(search_policy),
                      std::move(tune_option));
}

TVM_REGISTER_GLOBAL("ansor.TuneOption")
.set_body_typed([](int n_trials, int early_stopping,
                   int num_measure_per_round, int verbose, Builder builder,
                   Runner runner, Array<MeasureCallback> measure_callbacks,
                   Array<SearchCallback> pre_search_callbacks) {
  return TuneOption(n_trials, early_stopping, num_measure_per_round, verbose,
                    builder, runner, measure_callbacks, pre_search_callbacks);
});

TVM_REGISTER_GLOBAL("ansor.AutoScheduleBySearchTask")
.set_body_typed([](SearchTask task, SearchPolicy search_policy,
                   TuneOption tune_option) {
  te::Schedule sch;
  Array<te::Tensor> return_tensors;
  std::tie(sch, return_tensors) = AutoSchedule(task, search_policy, tune_option);

  return Array<ObjectRef>{sch, return_tensors};
});

TVM_REGISTER_GLOBAL("ansor.AutoScheduleByWorkloadKey")
.set_body_typed([](std::string workload_key, Target target,
                   Target target_host, SearchPolicy search_policy,
                   HardwareParams hardware_params, TuneOption tune_option) {
  te::Schedule sch;
  Array<te::Tensor> return_tensors;
  std::tie(sch, return_tensors) =
      AutoSchedule(workload_key, target, target_host, search_policy,
                   hardware_params, tune_option);

  return Array<ObjectRef>{sch, return_tensors};
});

}  // namespace ansor
}  // namespace tvm
