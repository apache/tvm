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
 * \file src/relay/collage/candidate_partition.cc
 * \brief A potential partition in the Collage search.
 */

#ifndef TVM_RELAY_COLLAGE_CANDIDATE_PARTITION_H_
#define TVM_RELAY_COLLAGE_CANDIDATE_PARTITION_H_

#include <tvm/runtime/container/string.h>
#include <tvm/target/compilation_config.h>

#include <memory>
#include <string>
#include <vector>

#include "./candidate_function_cache.h"
#include "./cost.h"
#include "./cost_estimator.h"
#include "./name_supply.h"
#include "./sub_graph.h"

namespace tvm {
namespace relay {
namespace collage {

class PartitionSpec;

/*!
 * \brief A candidate partition w.r.t. the overall Relay model.
 *
 * We represent the partition as a sub-graph. This means not only can we represent the scope
 * of Relay sub-expressions intended for a particular partition (or kernel), but we can also
 * represent various conventions for encoding how the operators within the partition should be
 * tagged for downstream processing.
 */
class CandidatePartitionNode : public Object {
 public:
  CandidatePartitionNode() = default;

  /*!
   * \brief Combination of all the partition rule names which produced this candidate.
   * For debugging and explainability.
   */
  String rule_name_;

  /*!
   * \brief The sub-graph of the overall expression matched by the partition rule.
   */
  SubGraph sub_graph_;

  /*!
   * \brief The partition specification which produced this candidate.
   */
  ObjectRef /* actually PartitionSpec */ spec_;

  /*!
   * \brief The (cached) cost of the partition.
   *
   * Initially Cost::Unknown, calculated and cached by EstimateCost.
   */
  mutable Cost cost_ = Cost::Unknown();

  void VisitAttrs(AttrVisitor* v);

  /*!
   * \brief Returns the partition specification which produced this candidate.
   */
  PartitionSpec partition_spec() const;

  /*!
   * \brief Returns the name of the partition specification which produced this candidate.
   */
  std::string partition_spec_name() const;

  /*!
   * \brief Returns the target of the partition specification which produced this candidate.
   */
  Target target() const;

  /*!
   * \brief Return the estimated cost of the candidate partition, using \p cost_estimator and
   * \p cache.
   */
  Cost EstimatedCost(const DataflowGraph& dataflow_graph, const CostEstimator& cost_estimator,
                     const std::shared_ptr<CandidateFunctionCache>& cache) const;

  /*!
   * \brief Returns a brief description of candidate suitable for debugging output.
   */
  std::string ToSummary(const DataflowGraph& dataflow_graph) const;

  std::string ToString() const;

  static constexpr const char* _type_key = "relay.collage.CandidatePartition";
  TVM_DECLARE_FINAL_OBJECT_INFO(CandidatePartitionNode, Object);
};

class CandidatePartition : public ObjectRef {
 public:
  CandidatePartition(String rule_name, SubGraph sub_graph,
                     ObjectRef /* actually PartitionSpec */ spec, Cost cost = Cost::Unknown());

  bool operator<(const CandidatePartition& that) const;

  /*!
   * \brief Returns true if this and \p that candidate are disjoint, have the same (or no) target,
   * and touch. This does not imply the \p DisjointUnion of this and that will be valid. For
   * example, the result may be too deep or have too many outputs.
   */
  bool AreTouching(const DataflowGraph& dataflow_graph, const CandidatePartition& that) const;

  /*!
   * \brief Returns the disjoint union of this and \p that.
   */
  CandidatePartition DisjointUnion(const DataflowGraph& dataflow_graph,
                                   const CandidatePartition& that) const;

  /*!
   * \brief Returns the disjoint union of all \p candidates.
   */
  static CandidatePartition DisjointUnion(const DataflowGraph& dataflow_graph,
                                          std::vector<CandidatePartition> candidates);

  /*!
   * \brief Returns the root expression of \p dataflow_graph rewritten to apply all the partitions
   * implied by \p candidates. The candidates can be in any order but must be disjoint.
   */
  static Expr ParallelRewrite(const DataflowGraph& dataflow_graph,
                              const std::vector<CandidatePartition>& candidates);

  /*!
   * Eagerly merge all touching candidates for the same target. The candidates must be disjoint
   * and have their Targets filled in. This is typically called on the optimal list of candidate
   * partitions found by the Collage search in order to remove unnecessary partition boundaries.
   * Ideally the search would never produce such candidates however to keep the search space
   * manageable Collage may only consider candidate partitions up to a particular depth.
   */
  static std::vector<CandidatePartition> MaxCoalesce(const DataflowGraph& dataflow_graph,
                                                     std::vector<CandidatePartition> candidates);

  TVM_DEFINE_OBJECT_REF_METHODS(CandidatePartition, ObjectRef, CandidatePartitionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CandidatePartitionNode);
};

CandidatePartition WithRuleName(CandidatePartition candidate, String rule_name);
CandidatePartition WithTarget(CandidatePartition candidate, Target target);
CandidatePartition WithSubGraph(CandidatePartition candidate, SubGraph sub_graph);

struct CandidatePartitionHash {
  size_t operator()(const CandidatePartition& candidate) const {
    return candidate->sub_graph_->hash();
  }
};

struct CandidatePartitionEquals {
  bool operator()(const CandidatePartition& left, const CandidatePartition& right) const {
    return *left->sub_graph_.get() == *right->sub_graph_.get();
  }
};

struct CandidatePartitionCompare {
  bool operator()(const CandidatePartition& left, const CandidatePartition& right) const {
    return *left->sub_graph_.get() < *right->sub_graph_.get();
  }
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_CANDIDATE_PARTITION_H_
