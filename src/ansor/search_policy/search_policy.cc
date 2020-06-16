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
 * \file ansor/search_policy/search_policy.cc
 * \brief The base class for search policy
 */

#include "search_policy.h"

#include <tvm/runtime/registry.h>

#include "../serialization.h"

namespace tvm {
namespace ansor {

TVM_REGISTER_OBJECT_TYPE(SearchPolicyNode);
TVM_REGISTER_OBJECT_TYPE(PreLoadMeasuredStatesNode);

void SearchPolicyNode::PreLoadMeasuredStates(const std::string& log_file) {
  LogReader reader = LogReaderNode::make(log_file);
  const auto& res = reader->ReadLines(-1);
  size_t log_size = res.first.size();
  CHECK_EQ(log_size, res.second.size());
  if (log_size) {
    std::vector<State> measured_states;
    std::vector<float> measured_throughputs;
    for (size_t i = 0; i < log_size; i++) {
      const auto& inp = res.first[i];
      if (inp->task->workload_key == cur_task_->workload_key &&
          inp->task->target->target_name.compare(
              cur_task_->target->target_name) == 0) {
        State state = cur_task_->compute_dag.GetInitState();
        state.CopyOnWrite()->transform_steps = inp->state->transform_steps;
        state.DoSteps(inp->state->transform_steps, cur_task_->compute_dag);
        measured_states.emplace_back(std::move(state));
        measured_throughputs.push_back(res.second[i]->error_no == 0 ?
            (1.0 / FloatArrayMean(res.second[i]->costs)) : 0.0);
      }
    }
    cur_task_->compute_dag.InferBound(&measured_states);
    for (size_t i = 0; i < measured_states.size(); i ++) {
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

    StdCout(verbose_) << "Measured States Set: " << measured_states_set_.size()
                      << " state hashes loaded from " << log_file
                      << " for " << cur_task_->workload_key << std::endl;
  } else {
    StdCout(verbose_) << "Measured States Set: no states found from "
                      << log_file << " for " << cur_task_->workload_key
                      << std::endl;
  }
}

void SearchPolicyNode::RunCallbacks(const Array<SearchCallback>& callbacks) {
  if (callbacks.defined() && callbacks.size()) {
    PrintTitle("Process search callbacks", verbose_);
    for (const auto& callback : callbacks) {
      callback->callback(this);
    }
  }
}

SearchCallback PreLoadMeasuredStatesNode::make(std::string filename) {
  auto node = make_object<PreLoadMeasuredStatesNode>();
  node->filename = std::move(filename);
  return SearchCallback(node);
}

void PreLoadMeasuredStatesNode::callback(SearchPolicyNode* policy) {
  policy->PreLoadMeasuredStates(filename);
}

// Search Policy
TVM_REGISTER_GLOBAL("ansor.SearchPolicyContinueSearchOneRound")
.set_body_typed([](SearchPolicy policy, SearchTask task, int num_measure,
                   int verbose, ProgramMeasurer measurer) {
  Array<MeasureInput> inputs;
  Array<MeasureResult> results;
  std::tie(inputs, results) = policy->ContinueSearchOneRound(task, num_measure,
      verbose, measurer);
  return Array<ObjectRef>{inputs, results};
});

TVM_REGISTER_GLOBAL("ansor.SearchPolicyRunCallbacks")
.set_body_typed([](SearchPolicy policy, Array<SearchCallback> callbacks) {
  policy->RunCallbacks(callbacks);
});

TVM_REGISTER_GLOBAL("ansor.SearchPolicySetTask")
.set_body_typed([](SearchPolicy policy, SearchTask task) {
  policy->cur_task_ = task;
});

TVM_REGISTER_GLOBAL("ansor.SearchPolicySetVerbose")
.set_body_typed([](SearchPolicy policy, int verbose) {
  policy->verbose_ = verbose;
});

TVM_REGISTER_GLOBAL("ansor.PreLoadMeasuredStates")
.set_body_typed([](std::string filename) {
  return PreLoadMeasuredStatesNode::make(filename);
});

}  // namespace ansor
}  // namespace tvm
