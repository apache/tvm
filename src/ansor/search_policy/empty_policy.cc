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

#include "empty_policy.h"

#include <tvm/runtime/registry.h>

namespace tvm {
namespace ansor {

TVM_REGISTER_NODE_TYPE(EmptyPolicyNode);

State EmptyPolicyNode::Search(SearchTask task, int n_trials, int early_stopping,
    int num_measure_per_iter, int verbose, ProgramMeasurer measurer,
    Array<SearchCallback> pre_search_callbacks) {
  cur_task = task;

  // Run pre_search_callbacks before the search process
  // This Interface is used to set some init status
  RunCallbacks(pre_search_callbacks);

  if (n_trials <= 1) {
    const auto& res = SearchOneRound();
    CHECK_GT(res.size(), 0);
    return res[0];
  } else {
    std::vector<MeasureInput> inputs;
    std::vector<MeasureResult> results;

    measurer->Reset();
    int ct = 0;
    while (ct < n_trials) {
      const auto& res = SearchOneRound();
      ct += res.size();
      inputs.clear();
      for (const auto& state : res) {
        inputs.emplace_back(cur_task, state);
      }
      measurer->Measure(cur_task, GetRef<SearchPolicy>(this), inputs, &results);
    }

    return measurer->best_state[cur_task->workload_key];
  }
}

std::pair<Array<MeasureInput>, Array<MeasureResult> > EmptyPolicyNode::ContinueSearchOneRound(
    SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) {
  std::vector<MeasureInput> inputs;
  std::vector<MeasureResult> results;

  const auto& res = SearchOneRound();
  for (const auto& state : res) {
    inputs.emplace_back(cur_task, state);
  }
  measurer->Measure(cur_task, GetRef<SearchPolicy>(this), inputs, &results);

  Array<MeasureInput> inputs_arr(std::make_move_iterator(inputs.begin()),
                                 std::make_move_iterator(inputs.end()));
  Array<MeasureResult> results_arr(std::make_move_iterator(results.begin()),
                                   std::make_move_iterator(results.end()));
  return std::make_pair(std::move(inputs_arr), std::move(results_arr));
}

std::vector<State> EmptyPolicyNode::SearchOneRound() {
  std::vector<State> res;
  res.push_back(cur_task->compute_dag.GetInitState());
  return res;
}

TVM_REGISTER_GLOBAL("ansor.EmptyPolicy")
.set_body_typed([]() { return EmptyPolicy(make_object<EmptyPolicyNode>()); });

}  // namespace ansor
}  // namespace tvm
