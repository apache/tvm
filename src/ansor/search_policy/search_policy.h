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
 * \brief The base class of search policies, including the abstract definition of search policy and
 * other supporting data structures.
 *
 * The basic schedule search process for Ansor is design to be:
 * `Program sampling` -> `Performance Tuning`.
 *
 * In `Program sampling`, we use some predefined precise or heuristic rules to generate several
 * initial schedules. Based on these initial starting points, we perform `Performance Tuning` which
 * uses cost model based evolutionary search to select schedules with the best performance.
 *
 * Candidate schedules are measured against the specific hardware target.
 *
 * \note Adding a new search policy.
 * In design, there's no need for users to implement their own search policy, our formal search
 * policy(will be brought later) should be enough to cover most use cases. Meanwhile, a custom rule
 * mechanism will be provided to enable user-defined template search to serve the same functionality
 * as the current AutoTVM template.
 *
 * This guide is to help understand it better and incase some advanced users have special
 * requirements.
 * 1. The only funcion that must be implemented is Search(), the design principe for it is to be
 * the entry of starting a schedule search process and returns the best schedule get.
 * 2. Information about the compute declaration of ops/subgraphs can be acquired from SearchTask.
 * This structure also contains some information about the target device. (e.g. knowing the weight
 * of the device vector unit, we can limit the max vectorize size during schedule generating)
 * 3. SearchCallback provides more flexibility to do extra affairs during the search process.
 * 4. ProgramMeasurer provides a simple but useful api to help check the performance of states get
 * during the search process.
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_
#define TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_

#include <tvm/node/node.h>

#include <unordered_set>
#include <vector>

#include "../search_task.h"

namespace tvm {
namespace ansor {

class ProgramMeasurer;
class SearchPolicyNode;

/*!
 * \brief Callback function to be called by the search process.
 * This interface allows to do extra initializations before schedule search or extra
 * check during/after the schedule search.
 */
class SearchCallbackNode : public Object {
 public:
  /*!
   * \brief Run the registered callback function.
   * \param policy A pointer to a SearchPolicyNode.
   */
  virtual void Callback(SearchPolicyNode* policy) = 0;

  static constexpr const char* _type_key = "ansor.SearchCallback";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchCallbackNode, Object);
};

/*!
 * \brief Managed reference to SearchCallbackNode.
 * \sa SearchCallbackNode
 */
class SearchCallback : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchCallback, ObjectRef, SearchCallbackNode);
};

/*!
 * \brief The base class of search policies.
 */
class SearchPolicyNode : public Object {
 public:
  /*! \brief The current search task. */
  SearchTask cur_task;
  /*!
   * \brief Verbose level to control the screen output during schedule search.
   * 0 for silent, 1 to output information.
   */
  int verbose;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("cur_task", &cur_task);
    v->Visit("verbose", &verbose);
  }

  /*!
   * \brief Do schedule search for a task. Takes the SearchTask as input and returns the best state
   * get during the search process.
   * \param task The target search task.
   * \param num_measure_trials Total schedules to be tried during this search.
   * \param early_stopping Early stop if no better schedule is found.
   * \param num_measures_per_round Max measure batch in one search round.
   * \param verbose Verbose level. 0 for silent, 1 to output information during schedule search.
   * \param measurer A ProgramMeasurer which packs ProgramBuilder & ProgramRunner inside.
   * \param pre_search_callbacks SearchCallback to be called before schedule search.
   * \return The best state get.
   */
  virtual State Search(SearchTask task, int num_measure_trials, int early_stopping,
                       int num_measures_per_round, int verbose, ProgramMeasurer measurer,
                       Array<SearchCallback> pre_search_callbacks) = 0;

  /*!
   * \brief Call SearchCallback with the current SearchPolicyNode
   * \param callbacks SearchCallback to be called.
   */
  void RunCallbacks(const Array<SearchCallback>& callbacks);

  static constexpr const char* _type_key = "ansor.SearchPolicy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchPolicyNode, Object);

 protected:
  /*!
   * \brief The set of already measured states.
   * We store the string format for redundancy check.
   */
  std::unordered_set<String> measured_states_set_;
  /*! \brief The array of already measured states. */
  std::vector<State> measured_states_vector_;
  /*! \brief The throughputs of already measured states */
  std::vector<float> measured_states_throughputs_;
};

/*!
 * \brief Managed reference to SearchPolicyNode.
 * \sa SearchPolicyNode
 */
class SearchPolicy : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SearchPolicy, ObjectRef, SearchPolicyNode);
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_
