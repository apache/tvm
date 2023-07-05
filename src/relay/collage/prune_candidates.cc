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
 * \file src/relay/collage/prune_candidates.cc
 * \brief Try to remove candidates which will never contribute to an optimal partitioning.
 */

#include "./prune_candidates.h"

#include "./dataflow_graph.h"
#include "./gather_partition_specs.h"

namespace tvm {
namespace relay {
namespace collage {

namespace {

/*!
 * \brief Returns a map from post-dfs dataflow node indices to the indices within \p candidates for
 * those candidates which intersect that dataflow node.
 *
 * NOTE: The index set in the vector results is over candidate indices not post-dfs indices!
 */
std::vector<IndexSet> MakeInsideMap(const DataflowGraph& dataflow_graph,
                                    const std::vector<CandidatePartition>& candidates) {
  std::vector<IndexSet> result(dataflow_graph.size(), IndexSet(candidates.size()));
  for (size_t i = 0; i < candidates.size(); ++i) {
    CandidatePartition candidate = candidates[i];
    for (PostDfsIndex index : candidate->sub_graph_->inside_) {
      result[index].Add(i);
    }
  }
  return result;
}

/*!
 * \brief Returns the maximal candidates within \p candidates. A candidate is maximal if it is not
 * contained by any super-candidate for the same target.
 */
std::vector<CandidatePartition> MaximalCandidates(
    const DataflowGraph& dataflow_graph, const std::vector<CandidatePartition>& candidates) {
  std::vector<IndexSet> inside_map = MakeInsideMap(dataflow_graph, candidates);
  std::vector<CandidatePartition> result;
  for (size_t i = 0; i < candidates.size(); ++i) {
    CandidatePartition maximal_candidate = candidates[i];
    bool has_super_candidate = false;
    IndexSet explored_candidates(candidates.size());  // over candidates!
    for (PostDfsIndex index : maximal_candidate->sub_graph_->inside_) {
      for (size_t j : inside_map[index]) {
        if (i == j) {
          // Ignore self.
          continue;
        }
        if (explored_candidates[j]) {
          // Already checked.
          continue;
        }
        explored_candidates.Add(j);
        CandidatePartition super_candidate = candidates[j];
        if (maximal_candidate->spec_ == super_candidate->spec_ &&
            maximal_candidate->sub_graph_->inside_.IsSubset(super_candidate->sub_graph_->inside_)) {
          has_super_candidate = true;
          break;
        }
      }
      if (has_super_candidate) {
        break;
      }
    }
    if (!has_super_candidate) {
      VLOG(2) << "Found maximal candidate " << maximal_candidate->ToString();
      result.emplace_back(maximal_candidate);
    }
  }
  VLOG(1) << "Have " << result.size() << " maximal candidates";
  return result;
}

/*!
 * \brief Returns all the candidates in \p candidates which intersect without being equal.
 */
std::vector<CandidatePartition> IntersectingCandidates(
    const DataflowGraph& dataflow_graph, const std::vector<CandidatePartition>& candidates) {
  std::vector<IndexSet> inside_map = MakeInsideMap(dataflow_graph, candidates);
  IndexSet intersecting(candidates.size());  // over candidates!
  for (size_t i = 0; i < candidates.size(); ++i) {
    CandidatePartition intersecting_candidate = candidates[i];
    IndexSet explored_candidates(candidates.size());  // over candidates!
    for (PostDfsIndex index : intersecting_candidate->sub_graph_->inside_) {
      for (size_t j : inside_map[index]) {
        if (j < i) {
          // Intersection is commutative.
          continue;
        }
        if (i == j) {
          // Ignore self.
          continue;
        }
        if (explored_candidates[j]) {
          // Already checked.
          continue;
        }
        explored_candidates.Add(j);
        CandidatePartition other_candidate = candidates[j];
        if (intersecting_candidate->sub_graph_->inside_ == other_candidate->sub_graph_->inside_) {
          // Have same inside set.
          continue;
        }
        VLOG(2) << "Candidate " << intersecting_candidate->ToString() << " intersects with "
                << other_candidate->ToString();
        intersecting.Add(i);
        intersecting.Add(j);
      }
    }
  }
  std::vector<CandidatePartition> result;
  for (size_t i : intersecting) {
    CandidatePartition candidate = candidates[i];
    VLOG(2) << "Found intersecting candidate " << candidate->ToString();
    result.emplace_back(candidate);
  }
  VLOG(1) << "Have " << result.size() << " intersecting candidates";
  return result;
}

/*!
 * \brief Returns the set operation left - right.
 */
std::vector<CandidatePartition> SetDifference(const std::vector<CandidatePartition>& left,
                                              const std::vector<CandidatePartition>& right) {
  std::unordered_set<CandidatePartition, CandidatePartitionHash, CandidatePartitionEquals>
      right_set(right.begin(), right.end());
  std::vector<CandidatePartition> result;
  for (const auto& candidate : left) {
    if (right_set.count(candidate) == 0) {
      result.emplace_back(candidate);
    }
  }
  return result;
}

/*!
 * \brief Adds everything in right to left. Returns the number of elements added.
 */
size_t SetUnionInPlace(
    std::unordered_set<CandidatePartition, CandidatePartitionHash, CandidatePartitionEquals>* left,
    const std::vector<CandidatePartition>& right) {
  size_t init_size = left->size();
  for (const auto& candidate : right) {
    left->emplace(candidate);
  }
  return left->size() - init_size;
}

}  // namespace

std::vector<CandidatePartition> PruneCandidates(
    const DataflowGraph& dataflow_graph,
    const std::vector<CandidatePartition>& initial_candidates) {
  VLOG_CONTEXT << "prune";
  // Start with all candidates available.
  std::vector<CandidatePartition> candidates = initial_candidates;
  std::unordered_set<CandidatePartition, CandidatePartitionHash, CandidatePartitionEquals> pruned;
  size_t initial_num_candidates = candidates.size();
  size_t num_rounds = 0;
  while (true) {
    VLOG_CONTEXT << "round " << ++num_rounds;
    VLOG(1) << "checking " << candidates.size() << " candidates";
    // Add all the maximal candidates to the pruned set.
    std::vector<CandidatePartition> maximal_candidates =
        MaximalCandidates(dataflow_graph, candidates);
    size_t num_new_pruned = SetUnionInPlace(&pruned, maximal_candidates);
    VLOG(1) << "Added " << num_new_pruned << " new pruned candidates";
    if (num_new_pruned == 0) {
      // We've reached a fixed point.
      break;
    }
    // If two pruned candidates intersect without being equal then we may miss valid
    // paths during search. So remove those intersecting candidates from the available candidates
    // and try again so as to find smaller candidates to 'bridge the gaps'.
    std::vector<CandidatePartition> pruned_vec(pruned.begin(), pruned.end());
    std::vector<CandidatePartition> intersecting_candidates =
        IntersectingCandidates(dataflow_graph, pruned_vec);
    // We need more maximal candidates to fill in the gaps between the current pruned candidates.
    // Force that by removing the intersecting candidates from the set of available candidates
    // and going around again.
    candidates = SetDifference(candidates, intersecting_candidates);
  }

  std::vector<CandidatePartition> result(pruned.begin(), pruned.end());
  // Re-establish a canonical order of candidates.
  std::sort(result.begin(), result.end());
  VLOG(1) << "Pruned " << initial_num_candidates - result.size() << " candidates (ie from "
          << initial_num_candidates << " to " << result.size() << ")";
  return result;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
