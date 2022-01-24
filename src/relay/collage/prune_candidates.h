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
 * \file src/relay/collage/prune_candidates.h
 * \brief Try to remove candidates which will never contribute to an optimal partitioning.
 */

#ifndef TVM_RELAY_COLLAGE_PRUNE_CANDIDATES_H_
#define TVM_RELAY_COLLAGE_PRUNE_CANDIDATES_H_

#include <vector>

#include "./candidate_partition.h"
#include "./dataflow_graph.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Returns \p initial_candidates with all unnecessary candidates pruned.
 *
 * We prune according to the following two heuristics:
 * 1. Given partitions (A, target) and (B, target) then
 *    cost(A union B, target) < cost(A, target) + cost(B, target).
 *    That is, there's no use estimating the cost of small partitions when a larger partition
 *    containing them is also available. More precisely, call a partition 'maximal' if it is
 *    not contained by any other partition for the same target. Then we want to prefer maximal
 *    candidates when searching.
 * 2. Given maximal partitions (A union B, target) and (A union B, target') where
 *    target != target', then min(cost(A union B, target), cost(A union B, target')) <
 *    min(cost(A, target) + cost(B, target'), cost(A, target') + cost(B, target)).
 *    That is, there's no use estimating cross-combinations of partitions which are not maximal.
 *
 * However, we can't prune a non-maximal candidate if it will make some other maximal candidate
 * unreachable during the Collage search. We achieve this by iterating until fixed point:
 *  - Find maximal candidates of current set of candidates.
 *  - Add those maximal candidates to the output 'pruned' set.
 *  - If any two candidates in the 'pruned' set intersect without being equal, remove those from
 *    the current set of candidates and go around again. That will force more candidates to
 *    be considered 'maximal'.
 * That over-approximates the true necessary candidates but is at least simple.
 *
 * CAUTION: This is pretty experimental. The above heuristics won't always be safe, and I don't
 * have a proof the pruned candidate set won't lead to 'No candidate was found covering
 * sub-expression...' errors in Partitioner::Partition().
 */
std::vector<CandidatePartition> PruneCandidates(
    const DataflowGraph& dataflow_graph, const std::vector<CandidatePartition>& initial_candidates);

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_PRUNE_CANDIDATES_H_
