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
 * \file tvm/auto_scheduler/search_policy.h
 * \brief The base class of search policies, including the abstract definition of search policy and
 * other supporting data structures.
 *
 * \note How to add a new search policy.
 * In design, there's no need for users to implement their own search policy, our formal search
 * policy(will be brought later) should be enough to cover most use cases. Meanwhile, a custom rule
 * mechanism will be provided to enable user-defined template search to serve the same functionality
 * as the current AutoTVM template.
 *
 * This guide is for advanced uses who have special requirements.
 * 1. The only function that must be implemented is Search(), which takes a task as input and
 * returns the best states found.
 * 2. Information about the compute declaration of ops/subgraphs can be acquired from SearchTask.
 * This structure also contains some information about the target device. (e.g. knowing the width
 * of the device vector unit, we can limit the max vectorize size during schedule search)
 * 3. SearchCallback provides more flexibility to do extra affairs before/after the search process.
 * 4. ProgramMeasurer provides a simple but useful api to help check the performance of states got
 * during the search process.
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_H_

#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/search_task.h>
#include <tvm/node/node.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace auto_scheduler {

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

  static constexpr const char* _type_key = "auto_scheduler.SearchCallback";
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

/*! \brief Preload measured states from a log file.
 * This can resume the state of the search policy */
class PreloadMeasuredStatesNode : public SearchCallbackNode {
 public:
  /*! \brief The name of the record log file. */
  String filename;

  void Callback(SearchPolicyNode* policy) final;

  static constexpr const char* _type_key = "auto_scheduler.PreloadMeasuredStates";
  TVM_DECLARE_FINAL_OBJECT_INFO(PreloadMeasuredStatesNode, SearchCallbackNode);
};

/*!
 * \brief Managed reference to PreloadMeasuredStatesNode.
 * \sa PreloadMeasuredStatesNode
 */
class PreloadMeasuredStates : public SearchCallback {
 public:
  /*!
   * \brief The constructor.
   * \param filename The name of the record log file.
   */
  explicit PreloadMeasuredStates(String filename);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PreloadMeasuredStates, SearchCallback,
                                        PreloadMeasuredStatesNode);
};

/*! \brief Attribute keys of ops used for SearchPolicy. */
struct SearchPolicyKey {
  /*! \brief Always apply unroll to the inner most iterator of the specificed iterators. */
  static constexpr const char* always_unroll_inner = "auto_scheduler_always_unroll_inner";
  /*! \brief The specified iterators will be placed in the inner most tile without split. */
  static constexpr const char* no_split_at_inner = "auto_scheduler_no_split_at_inner";
  /*! \brief The specified iterators are indices of const tensors in "fake reduction". */
  static constexpr const char* simplify_const_tensor_indices =
      "auto_scheduler_simplify_const_tensor_indices";
};

/*!
 * \brief The base class of search policies.
 */
class SearchPolicyNode : public Object {
 public:
  /*! \brief The current search task. */
  SearchTask search_task;
  /*!
   * \brief Verbose level to control the screen output during schedule search.
   * 0 for silent, 1 to output state & measure information during search process.
   */
  int verbose;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("search_task", &search_task);
    v->Visit("verbose", &verbose);
  }

  /*!
   * \brief Do schedule search for a task. Takes the SearchTask as input and returns the best state
   * found during the search.
   * \param num_measure_trials The number of total measurement trials.
   * \param early_stopping Stops the tuning early if no improvement after n measurements.
   * \param num_measures_per_round  The number of programs to be measured at each search round.
   * \param measurer A ProgramMeasurer to build and measure programs
   * \return The best state found.
   */
  virtual State Search(int num_measure_trials, int early_stopping, int num_measures_per_round,
                       ProgramMeasurer measurer) = 0;

  /*!
   * \brief Continue the search by doing an additional search round.
   * \param num_measure The number of measurements
   * \param measurer The measurer to measure programs
   * \return The measurement records for measurements in this search round
   */
  virtual std::pair<Array<MeasureInput>, Array<MeasureResult>> ContinueSearchOneRound(
      int num_measure, ProgramMeasurer measurer) = 0;

  /*!
   * \brief Preload measured states from a log file to resume the state of the search policy.
   * \param log_file The name of the record log file.
   */
  void PreloadMeasuredStates(const String& log_file);

  /*!
   * \brief Call SearchCallback with the current SearchPolicyNode
   * \param callbacks SearchCallback to be called.
   */
  void RunCallbacks(const Array<SearchCallback>& callbacks);

  static constexpr const char* _type_key = "auto_scheduler.SearchPolicy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchPolicyNode, Object);

 protected:
  /*!
   * \brief The set of already measured states.
   * We store the string format of a state for redundancy check. This is used to make sure a
   * measured state will never be measured again.
   */
  std::unordered_set<std::string> measured_states_set_;
  /*! \brief The array of already measured states.
   *  The good states can be used as the initial population in evolutionary search. */
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

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_H_
