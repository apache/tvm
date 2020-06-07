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

#include <tvm/node/node.h>
#include <vector>
#include <utility>
#include <string>
#include "../search_task.h"
#include "../measure.h"

namespace tvm {
namespace ansor {

class SearchPolicy;

/*! \brief The base class for search policy */
class SearchPolicyNode : public Object {
 public:
  virtual State Search(SearchTask task, int n_trials,
                       int early_stopping, int num_measure_per_iter,
                       int verbose, ProgramMeasurer measurer) = 0;

  virtual std::pair<Array<MeasureInput>, Array<MeasureResult> > ContinueSearchOneRound(
      SearchTask task, int num_measure, int verbose, ProgramMeasurer measurer) = 0;

  // Dict keys
  static constexpr const char* always_unroll_inner_key = "ansor_always_unroll_inner";
  static constexpr const char* always_unroll_key = "ansor_always_unroll";
  static constexpr const char* no_split_at_inner_key = "ansor_no_split_at_inner";
  static constexpr const char* no_split_at_outer_key = "ansor_no_split_at_outer";
  static constexpr const char* debug_skip_region_key = "ansor_debug_skip_region";
  static constexpr const char* last_split_is_one_key = "ansor_last_split_is_one";

  // Flag keys
  static constexpr const char* always_compute_inline_key = "ansor_always_compute_inline";
  static constexpr const char* no_cache_write_key = "ansor_no_cache_write";
  static constexpr const char* no_cache_read_key = "ansor_no_cache_read";
  static constexpr const char* tensor_core_support_key = "ansor_tensor_core_support";

  static constexpr const char *_type_key = "ansor.SearchPolicy";
  TVM_DECLARE_BASE_OBJECT_INFO(SearchPolicyNode, Object);
};
TVM_DEFINE_MUTABLE_OBJECT_REF(SearchPolicy, SearchPolicyNode);

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_SEARCH_POLICY_SEARCH_POLICY_H_
