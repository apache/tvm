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

#include "./candidate_partition_index.h"

#include "./gather_partition_specs.h"
#include "./prune_candidates.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

CandidatePartitionIndex::CandidatePartitionIndex(
    const std::unordered_map<const ExprNode*, VirtualDevice>* virtual_devices,
    DataflowGraph* dataflow_graph)
    : virtual_devices_(virtual_devices),
      dataflow_graph_(dataflow_graph),
      first_inside_index_to_candidates_(dataflow_graph->size()) {}

void CandidatePartitionIndex::Index(const Array<PartitionSpec>& partition_specs) {
  std::vector<CandidatePartition> candidates = Collect(partition_specs);
  candidates = PruneCandidates(*dataflow_graph_, candidates);
  // Index the candidates by their first inside index.
  for (auto& candidate : candidates) {
    first_inside_index_to_candidates_[candidate->sub_graph_->first_inside_index_].emplace_back(
        candidate);
  }
  size_ = candidates.size();
}

void CandidatePartitionIndex::EstimateAllCosts(
    const CostEstimator cost_estimator, const std::shared_ptr<CandidateFunctionCache>& cache) {
  size_t n = 0;
  for (PostDfsIndex index = 0; index < dataflow_graph_->size(); ++index) {
    for (const auto& candidate : first_inside_index_to_candidates_[index]) {
      LOG(INFO) << "Estimating cost of candidate " << candidate->ToSummary(*dataflow_graph_) << " ["
                << n++ << "/" << size_ << "]";
      // Cost will be cached in candidate as a side effect.
      Cost cost = candidate->EstimatedCost(*dataflow_graph_, cost_estimator, cache);
      LOG(INFO) << "Candidate has cost " << cost.ToString();
    }
  }
}

std::string CandidatePartitionIndex::ToSummary() const {
  std::vector<std::string> lines;
  for (const auto& candidates : first_inside_index_to_candidates_) {
    for (const auto& candidate : candidates) {
      if (candidate->partition_spec_name() == kHostSpecName) {
        continue;
      }
      lines.emplace_back(candidate->ToSummary(*dataflow_graph_));
    }
  }
  std::sort(lines.begin(), lines.end());
  std::ostringstream os;
  bool first = true;
  for (const auto& line : lines) {
    if (first) {
      first = false;
    } else {
      os << std::endl;
    }
    os << line;
  }
  return os.str();
}

bool CandidatePartitionIndex::IsCompatibleWithVirtualDevice(const CandidatePartition& candidate) {
  for (PostDfsIndex index : candidate->sub_graph_->inside_) {
    const ExprNode* sub_expr_node = dataflow_graph_->index_to_node(index)->node_ref_;
    if (sub_expr_node->IsInstance<OpNode>() || sub_expr_node->IsInstance<ConstructorNode>()) {
      // These nodes are target/device polymorphic.
      continue;
    }
    auto itr = virtual_devices_->find(sub_expr_node);
    ICHECK(itr != virtual_devices_->end()) << PrettyPrint(GetRef<Expr>(sub_expr_node));
    const Target& existing_target = itr->second->target;
    if (!existing_target.defined()) {
      // No constraint.
      continue;
    }
    if (StructuralEqual()(existing_target, candidate->target())) {
      // No disagreement.
      continue;
    }
    if (!candidate->target().IsExternalCodegenFor(itr->second->target)) {
      // The candidate's target is not an external codegen target compatible with the existing
      // target.
      // TODO(mbs): There's a conflict here between Collage's desire to leave some expression nodes
      // 'behind' on the VM and PlanDevice's desire to assign a primitive Target to every node.
      // I think PlanDevices is the one that needs to give here by leaving such nodes
      // unconstrained.
      VLOG(1) << "Ignoring candidate " << candidate->ToString()
              << " since incompatible with existing virtual device assignment of:" << std::endl
              << itr->second << std::endl
              << "to sub-graph:" << std::endl
              << PrettyPrint(GetRef<Expr>(sub_expr_node));
      return false;
    }
  }
  return true;
}

std::vector<CandidatePartition> CandidatePartitionIndex::Collect(
    const Array<PartitionSpec>& partition_specs) {
  VLOG_CONTEXT << "collecting";
  std::vector<CandidatePartition> result;
  for (const auto& spec : partition_specs) {
    VLOG_CONTEXT << "spec " << spec->spec_name_;
    VLOG(1) << "collecting candidates";
    std::vector<CandidatePartition> candidates = spec->AllCandidates(*dataflow_graph_);
    for (auto& candidate : candidates) {
      if (!IsCompatibleWithVirtualDevice(candidate)) {
        continue;
      }
      result.push_back(candidate);
    }
  }
  VLOG(1) << "Found " << result.size() << " candidates";
  return result;
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
