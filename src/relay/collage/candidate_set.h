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
 * \file src/relay/collage/candidate_set.h
 * \brief Collects a set of candidate partitions.
 */

#ifndef TVM_RELAY_COLLAGE_CANDIDATE_SET_H_
#define TVM_RELAY_COLLAGE_CANDIDATE_SET_H_

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./candidate_partition.h"
#include "./dataflow_graph.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Holds a vector of current candidates and the additions/removals to apply to them.
 */
struct CandidateSet {
  CandidateSet() = default;

  explicit CandidateSet(std::vector<CandidatePartition> candidates_to_add);

  /*!
   * \brief Schedule \p new_candidate for addition before the next round (unless it is not valid).
   */
  void Add(const DataflowGraph& dataflow_graph, const CandidatePartition& new_candidate);

  /*! \brief Schedule \p old_candidate for removal before the next round. */
  void Remove(const CandidatePartition& old_candidate);

  /*!
   * \brief Update \p current_candidates and \p first_new_index. Return false if no
   * new candidates were added, in which case we have reached a fixed point.
   */
  bool PrepareForNextRound();

  size_t size() const { return current_candidates_.size(); }

  CandidatePartition operator[](size_t i) const {
    ICHECK_LT(i, current_candidates_.size());
    return current_candidates_[i];
  }
  CandidatePartition at(size_t i) const { return (*this)[i]; }

  size_t first_new_index() const { return first_new_index_; }

  void sort() { std::sort(current_candidates_.begin(), current_candidates_.end()); }

  std::vector<CandidatePartition> MovedCurrentCandidates() {
    return std::move(current_candidates_);
  }

 private:
  /*!
   * \brief Index of first candidate in current_candidates added in last round. This can be used to
   * avoid considering candidates or candidate combinations which have already been considered in an
   * earlier round.
   */
  size_t first_new_index_ = 0;
  /*! \brief Candidates gathered in previous rounds. */
  std::vector<CandidatePartition> current_candidates_;
  /*! \brief New candidates gathered in the current round. */
  std::vector<CandidatePartition> candidates_to_add_;
  /*! \brief Existing candidates to remove before starting the next round. */
  std::vector<CandidatePartition> candidates_to_remove_;
  /*! \brief Which candidates have been seen so far and should not be added again. */
  std::unordered_set<CandidatePartition, CandidatePartitionHash, CandidatePartitionEquals> seen_;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_CANDIDATE_SET_H_
