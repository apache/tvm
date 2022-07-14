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
 * \file src/relay/collage/candidate_set.cc
 * \brief Collects a set of candidate partitions.
 */

#include "./candidate_set.h"

namespace tvm {
namespace relay {
namespace collage {

CandidateSet::CandidateSet(std::vector<CandidatePartition> candidates_to_add)
    : candidates_to_add_(std::move(candidates_to_add)) {
  for (const auto& candidate : candidates_to_add_) {
    seen_.emplace(candidate);
  }
}

void CandidateSet::Add(const DataflowGraph& dataflow_graph,
                       const CandidatePartition& new_candidate) {
  VLOG(2) << "adding " << new_candidate->ToString();
  if (seen_.count(new_candidate)) {
    VLOG(2) << "already seen candidate, ignoring";
    return;
  }
  seen_.emplace(new_candidate);
  candidates_to_add_.emplace_back(new_candidate);
}

void CandidateSet::Remove(const CandidatePartition& old_candidate) {
  ICHECK(seen_.count(old_candidate));
  VLOG(2) << "removing " << old_candidate->ToString();
  candidates_to_remove_.emplace_back(old_candidate);
}

bool CandidateSet::PrepareForNextRound() {
  size_t init_size = current_candidates_.size();
  for (const auto& candidate_to_remove : candidates_to_remove_) {
    current_candidates_.erase(
        std::remove(current_candidates_.begin(), current_candidates_.end(), candidate_to_remove),
        current_candidates_.end());
  }
  size_t num_removed = init_size - current_candidates_.size();
  candidates_to_remove_.clear();
  first_new_index_ = current_candidates_.size();
  for (const auto& new_candidate : candidates_to_add_) {
    current_candidates_.push_back(new_candidate);
  }
  size_t num_added = candidates_to_add_.size();
  candidates_to_add_.clear();
  VLOG(1) << "removed " << num_removed << " and added " << num_added << " candidates";
  return num_removed + num_added > 0;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
