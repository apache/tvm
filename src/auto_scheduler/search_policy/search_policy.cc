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
 * \file auto_scheduler/search_policy/search_policy.cc
 * \brief The base class of search policies.
 */

#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/auto_scheduler/search_policy.h>
#include <tvm/runtime/registry.h>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(SearchCallbackNode);
TVM_REGISTER_OBJECT_TYPE(SearchPolicyNode);
TVM_REGISTER_OBJECT_TYPE(PreloadMeasuredStatesNode);

void SearchPolicyNode::PreloadMeasuredStates(const String& log_file) {
  RecordReader reader = RecordReader(log_file);
  const auto& res = reader->ReadLines(-1);
  size_t log_size = res.first.size();
  ICHECK_EQ(log_size, res.second.size());
  if (log_size) {
    Array<State> measured_states;
    std::vector<float> measured_throughputs;
    for (size_t i = 0; i < log_size; i++) {
      const auto& inp = res.first[i];
      if (inp->task->workload_key == search_task->workload_key &&
          inp->task->target->kind->name.compare(search_task->target->kind->name) == 0) {
        State state = search_task->compute_dag->init_state;
        auto pstate = state.CopyOnWrite();
        pstate->transform_steps = inp->state->transform_steps;
        for (const auto& step : pstate->transform_steps) {
          StepApplyToState(step, &state, search_task->compute_dag);
        }
        measured_states.push_back(std::move(state));
        measured_throughputs.push_back(
            res.second[i]->error_no == 0 ? (1.0 / FloatArrayMean(res.second[i]->costs)) : 0.0);
      }
    }
    // We can assume the recorded states will all be valid after infer bound
    measured_states = search_task->compute_dag.InferBound(measured_states);
    for (size_t i = 0; i < measured_states.size(); i++) {
      auto& state = measured_states[i];
      const auto& state_str = state.ToStr();
      if (!measured_states_set_.count(state_str)) {
        measured_states_set_.insert(state_str);
        if (measured_throughputs[i] != 0.0) {
          measured_states_vector_.emplace_back(std::move(state));
          measured_states_throughputs_.emplace_back(measured_throughputs[i]);
        }
      }
    }

    StdCout(verbose) << "SearchPolicy: Loaded " << measured_states_set_.size()
                     << " measurement records from " << log_file << " for "
                     << search_task->workload_key << std::endl;
  } else {
    StdCout(verbose) << "SearchPolicy: No measurement records found in " << log_file << " for "
                     << search_task->workload_key << std::endl;
  }
}

void SearchPolicyNode::RunCallbacks(const Array<SearchCallback>& callbacks) {
  for (const auto& callback : callbacks) {
    callback->Callback(this);
  }
}

PreloadMeasuredStates::PreloadMeasuredStates(String filename) {
  auto node = make_object<PreloadMeasuredStatesNode>();
  node->filename = std::move(filename);
  data_ = std::move(node);
}

void PreloadMeasuredStatesNode::Callback(SearchPolicyNode* policy) {
  policy->PreloadMeasuredStates(filename);
}

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyRunCallbacks")
    .set_body_typed([](SearchPolicy policy, Optional<Array<SearchCallback>> callbacks) {
      if (callbacks) {
        policy->RunCallbacks(callbacks.value());
      }
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicyContinueSearchOneRound")
    .set_body_typed([](SearchPolicy policy, int num_measure, ProgramMeasurer measurer) {
      auto [inputs, results] = policy->ContinueSearchOneRound(num_measure, measurer);
      return Array<ObjectRef>{inputs, results};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SearchPolicySetVerbose")
    .set_body_typed([](SearchPolicy policy, int verbose) { policy->verbose = verbose; });

TVM_REGISTER_GLOBAL("auto_scheduler.PreloadMeasuredStates").set_body_typed([](String filename) {
  return PreloadMeasuredStates(filename);
});

}  // namespace auto_scheduler
}  // namespace tvm
