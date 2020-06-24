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
 * \file ansor/search_policy/empty_policy.h
 * \brief This is an basic example of search policy
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_EMPTY_POLICY_H_
#define TVM_ANSOR_SEARCH_POLICY_EMPTY_POLICY_H_

#include <utility>
#include <vector>

#include "search_policy.h"

namespace tvm {
namespace ansor {

/*!
 * \file ansor/search_policy/empty_policy.h
 * \brief This is an basic example for search policy. The EmptyPolicy will
 * always generates the init state of a ComputeDAG.
 */
class EmptyPolicyNode : public SearchPolicyNode {
 public:
  /*! \brief Search and make n_trails measurements.
   *  \returns the best state
   */
  State Search(SearchTask task, int n_trials,
               int early_stopping, int num_measure_per_iter,
               int verbose, ProgramMeasurer measurer,
               Array<SearchCallback> pre_search_callbacks) final;

  /*! \brief Continue search for one round. This is used by JointTuner
   * \returns the measurement pairs
   */
  std::pair<Array<MeasureInput>, Array<MeasureResult> > ContinueSearchOneRound(
      SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) final;

  static constexpr const char *_type_key = "ansor.EmptyPolicy";
  TVM_DECLARE_FINAL_OBJECT_INFO(EmptyPolicyNode, SearchPolicyNode);

 private:
  /*!
   * \brief Usually we need a sub function to generate several candidate states in each
   * search round.
   * \returns Several generated states
   */
  std::vector<State> SearchOneRound();
};

/*!
 * \brief Managed reference to EmptyPolicyNode.
 * \sa EmptyPolicyNode
 */
class EmptyPolicy : public SearchPolicy {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(EmptyPolicy, SearchPolicy, EmptyPolicyNode);
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_EMPTY_POLICY_H_