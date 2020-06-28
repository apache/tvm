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
 * \brief The base class for search policy, including the abstract defination of search policy and
 * some other supporting structures.
 * 
 * \note Adding a new search policy.
 * In design, there's no need for users to implement their own search policy, our formal search
 * policy(will be brought later) should be enough to cover auto schedule generation for different
 * ops/subgraphs, and in the meantime, a custom rule mechanism will be provided to enable
 * user-defined template search. (which should play a same role as the current AutoTVM template)
 * This guide is to help understand it better and incase some advanced users have special
 * requirements.
 * 1. The only funcion that must be implemented is Search(), the design principe for it is to be
 * the entry of starting a schedule search and returns the best schedule get.
 * 2. Imformations about the target ops/subgraphs can be acquired from SearchTask, this structure
 * also contains HardwareParams which can be used to limit the search space. (For exp. limit the
 * max vectorize size depending on the vector unit weight of a specific device)
 * 3. SearchCallback provides more flexibility to do extra affairs during the search process.
 * 4. ProgramMeasurer provides a simple but useful api to help check the performance of states get
 * during the search process.
 */

#ifndef TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_
#define TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_

#include <tvm/node/node.h>

#include <unordered_set>
#include <vector>
#include <utility>
#include <string>

#include "../search_task.h"

namespace tvm {
namespace ansor {

class ProgramMeasurer; class SearchPolicyNode;

/*!
 * \brief Callback function to be called by the search process.
 * This interface allows to do extra initializations before schedule search or extra
 * check during/after the schedule search.
 */
class SearchCallbackNode : public Object {
 public:
  /*!
   * \brief Run the registered callback function.
   * \param policy A pointer to SearchPolicyNode.
   */
  virtual void Callback(SearchPolicyNode* policy) = 0;

  static constexpr const char *_type_key = "ansor.SearchCallback";
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
 * \brief The base class for search policy.
 */
class SearchPolicyNode : public Object {
 public:
  /*! \brief The current search task. */
  SearchTask cur_task;
  /*!
   * \brief Verbose level to control the screen output during schedule search.
   * (0 means silent)
   */
  int verbose;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("cur_task", &cur_task);
    v->Visit("verbose", &verbose);
  }

  /*!
   * \brief Do schedule search for a task.
   * \param task The target search task.
   * \param n_trials Total schedules to be tried during this search.
   * \param early_stopping Early stop if no better schedule is found.
   * \param num_measure_per_round Max measure batch in one search round.
   * \param verbose Verbose level. (0 means silent)
   * \param measurer A ProgramMeasurer which packs Builder & Runner inside.
   * \param pre_search_callbacks SearchCallback to be called before schedule search.
   * \return The best state get.
   */
  virtual State Search(SearchTask task, int n_trials,
                       int early_stopping, int num_measure_per_round,
                       int verbose, ProgramMeasurer measurer,
                       Array<SearchCallback> pre_search_callbacks) = 0;

  /*!
   * \brief Call SearchCallback with the current SearchPolicyNode.u
   * \param callbacks SearchCallback to be called.
   */
  void RunCallbacks(const Array<SearchCallback>& callbacks);

  static constexpr const char *_type_key = "ansor.SearchPolicy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchPolicyNode, Object);

 protected:
  /*!
   * \brief The set of already measured states.
   * We store the string format for redundancy check.
   */
  std::unordered_set<std::string> measured_states_set_;
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
