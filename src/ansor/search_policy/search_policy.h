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
 * \file ansor/search_policy/search_policy.h
 * \brief The base class for search policy
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_
#define TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_

#include "../search_task.h"
#include <tvm/node/node.h>
#include <unordered_set>
#include <vector>
#include <utility>
#include <string>
#include "../measure.h"

namespace tvm {
namespace ansor {

class SearchPolicyNode;

/*! \brief Callback function to be called before or after the search process */
class SearchCallbackNode : public Object {
 public:
  virtual void callback(SearchPolicyNode* policy) = 0;

  static constexpr const char *_type_key = "ansor.SearchCallback";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchCallbackNode, Object);
};
TVM_DEFINE_MUTABLE_OBJECT_REF(SearchCallback, SearchCallbackNode);

/*! \brief The base class for search policy */
class SearchPolicyNode : public Object {
 public:
  SearchTask cur_task;   // The current task
  int verbose;           // Verbose level (0 means silent)

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("cur_task", &cur_task);
    v->Visit("verbose", &verbose);
  }

  // Search for a task
  virtual State Search(SearchTask task, int n_trials,
                       int early_stopping, int num_measure_per_iter,
                       int verbose, ProgramMeasurer measurer,
                       Array<SearchCallback> pre_search_callbacks) = 0;

  // Continue search one round for a task.
  // This is used in the task scheduler for searching for multiple tasks together.
  virtual std::pair<Array<MeasureInput>, Array<MeasureResult> > ContinueSearchOneRound(
      SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) = 0;

  // Run a list of callback functions
  void RunCallbacks(const Array<SearchCallback>& callbacks);

  static constexpr const char *_type_key = "ansor.SearchPolicy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchPolicyNode, Object);

 protected:
  // The set of the already measured states.
  // We store the string format for redundancy check
  std::unordered_set<std::string> measured_states_set_;
  // The array of already measured states.
  std::vector<State> measured_states_vector_;
  // The throughputs of already measured states
  std::vector<float> measured_states_throughputs_;
};
TVM_DEFINE_MUTABLE_OBJECT_REF(SearchPolicy, SearchPolicyNode);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_
