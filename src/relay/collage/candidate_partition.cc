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

#include "./candidate_partition.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/transform.h>

#include "../transforms/compiler_function_utils.h"
#include "./candidate_function_cache.h"
#include "./candidate_set.h"
#include "./partition_rule.h"
#include "./partition_spec.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

TVM_REGISTER_NODE_TYPE(CandidatePartitionNode);

void CandidatePartitionNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("rule_name", &rule_name_);
  v->Visit("sub_graph", &sub_graph_);
  v->Visit("spec", &spec_);
  // TODO(mbs): cost_
}

PartitionSpec CandidatePartitionNode::partition_spec() const {
  return Downcast<PartitionSpec>(spec_);
}

std::string CandidatePartitionNode::partition_spec_name() const {
  return Downcast<PartitionSpec>(spec_)->spec_name_;
}

Target CandidatePartitionNode::target() const { return Downcast<PartitionSpec>(spec_)->target_; }

std::string CandidatePartitionNode::ToSummary(const DataflowGraph& dataflow_graph) const {
  std::ostringstream os;
  os << sub_graph_->label_;
  os << " | (";
  bool first = true;
  for (PostDfsIndex index : sub_graph_->input_) {
    Expr sub_expr = dataflow_graph.index_to_node(index)->ref();
    if (CanInline(sub_expr)) {
      continue;
    }
    if (first) {
      first = false;
    } else {
      os << ", ";
    }
    os << PrettyPrint(sub_expr->checked_type());
  }
  os << ") -> (";
  first = true;
  for (PostDfsIndex index : sub_graph_->exit_) {
    Expr sub_expr = dataflow_graph.index_to_node(index)->ref();
    if (CanInline(sub_expr)) {
      continue;
    }
    if (first) {
      first = false;
    } else {
      os << ", ";
    }
    os << PrettyPrint(sub_expr->checked_type());
  }
  os << ") | ";
  os << sub_graph_->inside_.ToString();
  os << " | ";
  os << partition_spec_name();
  os << " | ";
  os << cost_.ToString();
  return os.str();
}

std::string CandidatePartitionNode::ToString() const {
  std::ostringstream os;
  os << "{rule_name=" << rule_name_;
  os << ",sub_graph=" << sub_graph_->ToString();
  os << ",spec_name=" << partition_spec_name();
  if (!cost_.is_unknown()) {
    os << ",cost=" << cost_.ToString();
  }
  os << "}";
  return os.str();
}

namespace {
/*!
 * \brief If function's body is a call to an inlined "Primitive" function, return it.
 * Otherwise return function directly.
 */
Function GetPrimitiveFunction(const Function& function) {
  if (const auto* call_node = function->body.as<CallNode>()) {
    if (const auto* function_node = call_node->op.as<FunctionNode>()) {
      if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
        return GetRef<Function>(function_node);
      }
    }
  }
  return function;
}

/*!
 * \brief Eta-expand any tuple arguments of \p function. Ie rewrite:
 * \code
 *   f(x: (t1, t2)) { ... x ... }
 * \endcode
 * to
 * \code
 *   f(x_1: t1, x_2: t2) { ... (x_1, x_2) ... }
 * \endcode
 */
Function EtaExpandTuples(const Function& function) {
  Map<Var, Expr> subst;
  Array<Var> new_params;
  for (const auto& param : function->params) {
    std::vector<TensorType> tensor_types = FlattenTupleType(param->type_annotation);
    if (tensor_types.size() == 1) {
      new_params.push_back(param);
    } else {
      Array<Expr> fields;
      for (size_t i = 0; i < tensor_types.size(); ++i) {
        Var new_param(param->name_hint() + "_" + std::to_string(i), tensor_types[i], param->span);
        new_param->checked_type_ = tensor_types[i];
        new_params.push_back(new_param);
        fields.push_back(new_param);
      }
      Tuple new_tuple(fields);
      subst.Set(param, new_tuple);
    }
  }
  if (subst.empty()) {
    return function;
  }
  return WithFields(function, new_params, Bind(function->body, subst));
}

}  // namespace

Cost CandidatePartitionNode::EstimatedCost(
    const DataflowGraph& dataflow_graph, const CostEstimator& cost_estimator,
    const std::shared_ptr<CandidateFunctionCache>& cache) const {
  if (cost_.is_unknown()) {
    VLOG_CONTEXT << "spec " << partition_spec_name();
    Function extracted_function = sub_graph_->ExtractAsFunction(dataflow_graph);
    VLOG(2) << "Extracted function:" << std::endl << PrettyPrint(extracted_function);
    extracted_function = EtaExpandTuples(extracted_function);
    VLOG(2) << "Validating function:" << std::endl << PrettyPrint(extracted_function);
    String error = partition_spec()->validate_sub_graph_func_(extracted_function);
    if (!error.empty()) {
      cost_ = Cost::Invalid();
      VLOG(1) << "Unable to rewrite function: " << error;
    } else {
      // The extracted function may be the eta-expansion of a "Primitive" function.
      // If so we want the cached external name and cost to be w.r.t. that function
      // rather than the outer so that we'll get a cache hit when we outline functions
      // in the final program.
      Function primitive_function = GetPrimitiveFunction(extracted_function);
      CandidateFunctionCache::Entry& entry =
          cache->GetEntry(sub_graph_->label_, primitive_function);
      if (entry.cost.is_unknown()) {
        IRModule mod = IRModule::FromExpr(extracted_function);
        VLOG(1) << "Outlining:" << std::endl << PrettyPrint(mod);
        mod = OutlineCompilerFunctions(cache)(mod);
        VLOG(1) << "Estimating cost of:" << std::endl
                << PrettyPrint(mod) << std::endl
                << "using target " << target()->ToDebugString();
        entry.cost = cost_estimator->Estimate(mod, target());
        VLOG(1) << "Measured cost as " << entry.cost.ToString();
      } else {
        VLOG(1) << "Reusing cost " << entry.cost.ToString()
                << " cached in candidate function cache";
      }
      cost_ = entry.cost;
    }
  } else {
    VLOG(1) << "Reusing cost " << cost_.ToString() << " cached in candidate";
  }
  return cost_;
}

CandidatePartition::CandidatePartition(String rule_name, SubGraph sub_graph,
                                       ObjectRef /* actually PartitionSpec */ spec, Cost cost) {
  auto node = runtime::make_object<CandidatePartitionNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_graph_ = std::move(sub_graph);
  node->spec_ = std::move(spec);
  node->cost_ = cost;
  data_ = std::move(node);
}

CandidatePartition WithRuleName(CandidatePartition candidate, String rule_name) {
  if (rule_name == candidate->rule_name_) {
    return candidate;
  }
  auto* node = candidate.CopyOnWrite();
  node->rule_name_ = std::move(rule_name);
  return GetRef<CandidatePartition>(node);
}

CandidatePartition WithSubGraph(CandidatePartition candidate, SubGraph sub_graph) {
  if (sub_graph == candidate->sub_graph_) {
    return candidate;
  }
  auto* node = candidate.CopyOnWrite();
  node->sub_graph_ = std::move(sub_graph);
  return GetRef<CandidatePartition>(node);
}

bool CandidatePartition::operator<(const CandidatePartition& that) const {
  // Order lexicographically on sub-graphs.
  if (*get()->sub_graph_.get() < *that->sub_graph_.get()) {
    return true;
  }
  if (*that->sub_graph_.get() < *get()->sub_graph_.get()) {
    return false;
  }
  // Break ties by rule name.
  return get()->rule_name_ < that->rule_name_;
}

bool CandidatePartition::AreTouching(const DataflowGraph& dataflow_graph,
                                     const CandidatePartition& that) const {
  return get()->spec_ == that->spec_ &&
         get()->sub_graph_.AreTouching(dataflow_graph, that->sub_graph_);
}

CandidatePartition CandidatePartition::DisjointUnion(const DataflowGraph& dataflow_graph,
                                                     const CandidatePartition& that) const {
  ICHECK_EQ(get()->spec_, that->spec_);
  return CandidatePartition(UnionLabels(get()->rule_name_, that->rule_name_),
                            get()->sub_graph_.DisjointUnion(dataflow_graph, that->sub_graph_),
                            get()->spec_, get()->cost_ + that->cost_);
}

/*static*/
CandidatePartition CandidatePartition::DisjointUnion(const DataflowGraph& dataflow_graph,
                                                     std::vector<CandidatePartition> candidates) {
  ICHECK_GT(candidates.size(), 1);
  CandidatePartition result = candidates.front();
  for (size_t i = 1; i < candidates.size(); ++i) {
    result = result.DisjointUnion(dataflow_graph, candidates[i]);
  }
  return result;
}

/*static*/
Expr CandidatePartition::ParallelRewrite(const DataflowGraph& dataflow_graph,
                                         const std::vector<CandidatePartition>& candidates) {
  std::vector<SubGraph> sub_graphs;
  sub_graphs.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    sub_graphs.emplace_back(candidate->sub_graph_);
  }
  return SubGraph::ParallelRewrite(dataflow_graph, sub_graphs);
}

/*static*/
std::vector<CandidatePartition> CandidatePartition::MaxCoalesce(
    const DataflowGraph& dataflow_graph, std::vector<CandidatePartition> candidates) {
  VLOG(1) << "Running MaxCoalesce over " << candidates.size() << " candidates";
  // This is an eager version of using the simple (kOpaque, kOpaque) combiner.

  // Switch to set representation.
  CandidateSet result_set(std::move(candidates));

  // Until fixed point...
  size_t num_rounds = 0;
  while (result_set.PrepareForNextRound()) {
    VLOG_CONTEXT << "round " << ++num_rounds;
    VLOG(1) << "checking " << result_set.size() << " candidates (" << result_set.first_new_index()
            << " existing)";
    IndexSet removed_this_round(result_set.size());  // over candidate indexes!

    // Build map from post-dfs indices to the indices of candidates with corresponding entry node.
    // NOTE: the index set is over candidate indices not post-dfs indices!
    std::vector<IndexSet> entry_map(dataflow_graph.size(), IndexSet(result_set.size()));
    for (size_t i = 0; i < result_set.size(); ++i) {
      CandidatePartition candidate = result_set.at(i);
      for (PostDfsIndex entry_index : candidate->sub_graph_->entry_) {
        entry_map[entry_index].Add(i);
      }
    }

    for (size_t i = 0; i < result_set.size(); ++i) {
      if (removed_this_round[i]) {
        // Already merged.
        continue;
      }
      CandidatePartition upstream = result_set.at(i);
      // Narrow our search to just those candidates which could touch.
      IndexSet possible_downstream(result_set.size());  // over candidate indexes!
      for (PostDfsIndex output_index : upstream->sub_graph_->output_) {
        possible_downstream = possible_downstream | entry_map[output_index];
      }
      for (size_t j : possible_downstream) {
        if (removed_this_round[j]) {
          // Already merged.
          continue;
        }
        if (i == j) {
          // Ignore self.
          continue;
        }
        CandidatePartition downstream = result_set.at(j);
        if (!upstream.AreTouching(dataflow_graph, downstream)) {
          continue;
        }
        CandidatePartition new_candidate = upstream.DisjointUnion(dataflow_graph, downstream);
        VLOG(2) << "Merging upstream candidate " << upstream->ToString()
                << " and downstream candidate " << downstream->ToString() << " to yield "
                << new_candidate->ToString();
        result_set.Add(dataflow_graph, new_candidate);
        result_set.Remove(upstream);
        removed_this_round.Add(i);
        result_set.Remove(downstream);
        removed_this_round.Add(j);
      }
    }
  }

  // Restore canonical order.
  result_set.sort();

  VLOG(1) << "MaxCoalesce produced " << result_set.size() << " candidates";
  return result_set.MovedCurrentCandidates();
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
