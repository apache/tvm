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
 * \file auto_scheduler/search_policy/empty_policy.h
 * \brief A simple example of the search policy which always returns the initial naive schedule
 * (state).
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_EMPTY_POLICY_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_EMPTY_POLICY_H_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/search_policy.h>

#include <utility>

namespace tvm {
namespace auto_scheduler {

/*!
 * \brief A simple example of the search policy which always returns the initial naive schedule
 * (state).
 * The key implementation for this structure is `Search()`, check `empty_policy.cc` for more
 * details.
 */
class EmptyPolicyNode : public SearchPolicyNode {
 public:
  State Search(int num_measure_trials, int early_stopping, int num_measures_per_round,
               ProgramMeasurer measurer) final;

  std::pair<Array<MeasureInput>, Array<MeasureResult>> ContinueSearchOneRound(
      int num_measure, ProgramMeasurer measurer) final;

  static constexpr const char* _type_key = "auto_scheduler.EmptyPolicy";
  TVM_DECLARE_FINAL_OBJECT_INFO(EmptyPolicyNode, SearchPolicyNode);

 private:
  /*!
   * \brief Use a sub function to generate several candidate states in each search round.
   * \returns The generated states
   */
  Array<State> SearchOneRound();
};

/*!
 * \brief Managed reference to EmptyPolicyNode.
 * \sa EmptyPolicyNode
 */
class EmptyPolicy : public SearchPolicy {
 public:
  explicit EmptyPolicy(SearchTask task, Optional<Array<SearchCallback>> init_search_callbacks);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EmptyPolicy, SearchPolicy, EmptyPolicyNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_EMPTY_POLICY_H_
