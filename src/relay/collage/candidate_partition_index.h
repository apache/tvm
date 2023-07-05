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
 * \file relay/collage/candidate_partition_index.h
 * \brief Index for finding relevant candidate partitions for a particular search state.
 */
#ifndef TVM_RELAY_COLLAGE_CANDIDATE_PARTITION_INDEX_H_
#define TVM_RELAY_COLLAGE_CANDIDATE_PARTITION_INDEX_H_

#include <tvm/relay/expr.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "./partition_spec.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Collects and indexes all the candidate partitions for the overall expression. This index
 * is used during partitioning search to find the next valid candidate partition to explore from the
 * current search state. We do not yet attempt to estimate the cost of each candidate partition, and
 * when we do so during the search we may discover it to be infeasible.
 */
class CandidatePartitionIndex {
 public:
  CandidatePartitionIndex(const std::unordered_map<const ExprNode*, VirtualDevice>* virtual_devices,
                          DataflowGraph* dataflow_graph);

  /*! \brief Constructs the index. */
  void Index(const Array<PartitionSpec>& partition_specs);

  /*! \brief Returns all the candidates which may begin at \p index. */
  const std::vector<CandidatePartition>& candidates_at(PostDfsIndex index) const {
    ICHECK_LT(index, dataflow_graph_->size());
    return first_inside_index_to_candidates_[index];
  }

  /*! \brief Estimates the casts of all candidates in the index. Each candidate caches its cost. */
  void EstimateAllCosts(const CostEstimator cost_estimator,
                        const std::shared_ptr<CandidateFunctionCache>& cache);

  size_t size() const { return size_; }

  std::string ToSummary() const;

 private:
  /*!
   * \brief Returns true if \p candidate's desired target is compatible with any existing target
   * constraints on the candidate's sub-expressions.
   */
  bool IsCompatibleWithVirtualDevice(const CandidatePartition& candidate);

  /*! \brief Returns all valid candidates found from \p partition_specs. */
  std::vector<CandidatePartition> Collect(const Array<PartitionSpec>& partition_specs);

  /*!
   * \brief The \p VirtualDevice for every sub-expression in the overall expression. Needed to
   * ensure candidates do not contradict the target/device placement already determined by
   * device planning.
   */
  const std::unordered_map<const ExprNode*, VirtualDevice>* virtual_devices_;

  /*! \brief Dataflow graph for overall expression. */
  DataflowGraph* dataflow_graph_;

  /*!
   * \brief Maps post-dfs indexes to the all the candidates which have that as their first inside
   * index, and which should be considered in the Collage search.
   */
  std::vector<std::vector<CandidatePartition>> first_inside_index_to_candidates_;

  /*! \brief Number of entries in above. */
  size_t size_ = 0;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_CANDIDATE_PARTITION_INDEX_H_
