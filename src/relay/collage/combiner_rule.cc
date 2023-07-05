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
 * \file src/relay/collage/combiner_rule.cc
 * \brief Helpers for the \p CombinePartitionRule
 */

#include "./combiner_rule.h"

#include "./partition_spec.h"

namespace tvm {
namespace relay {
namespace collage {

TVM_REGISTER_NODE_TYPE(SimpleCombinerRuleNode);

void SimpleCombinerRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

bool SimpleCombinerRuleNode::Fires(const DataflowGraph& dataflow_graph,
                                   const CandidatePartition& upstream,
                                   const CandidatePartition& downstream) const {
  return false;
}

std::string SimpleCombinerRuleNode::ToString() const {
  return "SimpleCombinerRule(" + rule_name_ + ")";
}

SimpleCombinerRule::SimpleCombinerRule(String rule_name) {
  auto node = runtime::make_object<SimpleCombinerRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(ByKindSimpleCombinerRuleNode);

void ByKindSimpleCombinerRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

bool ByKindSimpleCombinerRuleNode::Fires(const DataflowGraph& dataflow_graph,
                                         const CandidatePartition& upstream,
                                         const CandidatePartition& downstream) const {
  return upstream->sub_graph_->kind_ <= upstream_kind_ &&
         downstream->sub_graph_->kind_ <= downstream_kind_;
}

std::string ByKindSimpleCombinerRuleNode::ToString() const {
  std::ostringstream os;
  os << "ByKindSimpleCombinerRule(" << rule_name_ << ")";
  return os.str();
}

ByKindSimpleCombinerRule::ByKindSimpleCombinerRule(OpPatternKind upstream_kind,
                                                   OpPatternKind downstream_kind) {
  auto node = runtime::make_object<ByKindSimpleCombinerRuleNode>();
  String rule_name = KindToString(upstream_kind) + "->" + KindToString(downstream_kind);
  node->rule_name_ = std::move(rule_name);
  node->upstream_kind_ = upstream_kind;
  node->downstream_kind_ = downstream_kind;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(CombinerRuleNode);

void CombinerRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

void CombinerRuleNode::AppendAllResults(AppendAllResultsContext* ctxt) const {}

std::string CombinerRuleNode::ToString() const { return "CombinerRuleNode(" + rule_name_ + ")"; }

CombinerRule::CombinerRule(String rule_name) {
  auto node = runtime::make_object<CombinerRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AllSimpleCombinerRuleNode);

void AllSimpleCombinerRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

void AllSimpleCombinerRuleNode::AppendAllResults(AppendAllResultsContext* ctxt) const {
  VLOG(1) << "running AllSimpleCombinerRule(" << rule_name_ << ")";
  // Build map from post-dfs indices to the indices of candidates with corresponding entry node.
  // NOTE: the index set is over candidate indices not post-dfs indices!
  std::vector<IndexSet> entry_map(ctxt->dataflow_graph->size(),
                                  IndexSet(ctxt->candidate_set->size()));
  for (size_t i = 0; i < ctxt->candidate_set->size(); ++i) {
    CandidatePartition candidate = ctxt->candidate_set->at(i);
    for (PostDfsIndex entry_index : candidate->sub_graph_->entry_) {
      entry_map[entry_index].Add(i);
    }
  }

  for (size_t i = 0; i < ctxt->candidate_set->size(); ++i) {
    CandidatePartition upstream = ctxt->candidate_set->at(i);
    // Narrow our search to just those candidates which could touch.
    IndexSet possible_downstream(ctxt->candidate_set->size());
    for (PostDfsIndex output_index : upstream->sub_graph_->output_) {
      possible_downstream = possible_downstream | entry_map[output_index];
    }
    size_t start_j =
        i < ctxt->candidate_set->first_new_index() ? ctxt->candidate_set->first_new_index() : 0;
    for (size_t j : possible_downstream) {
      if (i == j) {
        continue;
      }
      if (i < start_j) {
        // We already explored the cross-product of candidates [0, first_new_index), so don't
        // do it again.
        continue;
      }
      // Note that the rules are not commutative so we can't just ignore if j < i.
      CandidatePartition downstream = ctxt->candidate_set->at(j);
      if (ctxt->max_depth > 0 &&
          upstream->sub_graph_->depth_ + downstream->sub_graph_->depth_ > ctxt->max_depth) {
        continue;
      }
      if (!upstream.AreTouching(*ctxt->dataflow_graph, downstream)) {
        continue;
      }
      for (const auto& simple_rule : simple_rules_) {
        if (simple_rule->Fires(*ctxt->dataflow_graph, upstream, downstream)) {
          CandidatePartition new_candidate =
              upstream.DisjointUnion(*ctxt->dataflow_graph, downstream);
          VLOG(2) << "Fired " << simple_rule->rule_name_ << " on upstream candidate "
                  << upstream->ToString() << " and downstream candidate " << downstream->ToString()
                  << " to yield " << new_candidate->ToString();
          ctxt->candidate_set->Add(*ctxt->dataflow_graph, new_candidate);
        }
      }
    }
  }
}

std::string AllSimpleCombinerRuleNode::ToString() const {
  std::ostringstream os;
  os << "AllSimpleCombinerRule(" << rule_name_;
  for (const auto& simple : simple_rules_) {
    os << ", " << simple->ToString();
  }
  os << ")";
  return os.str();
}

AllSimpleCombinerRule::AllSimpleCombinerRule(String rule_name,
                                             Array<SimpleCombinerRule> simple_rules) {
  auto node = runtime::make_object<AllSimpleCombinerRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->simple_rules_ = std::move(simple_rules);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(TupleArgCombinerRuleNode);

void TupleArgCombinerRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

void TupleArgCombinerRuleNode::AppendAllResults(AppendAllResultsContext* ctxt) const {
  VLOG(1) << "running TupleArgCombinerRule(" << rule_name_ << ")";
  // Build map from post-dfs index to the indices of injective candidates with corresponding entry
  // node. NOTE: the index set is over candidate indices not post-dfs indices!
  std::vector<IndexSet> exit_map(ctxt->dataflow_graph->size(),
                                 IndexSet(ctxt->candidate_set->size()));
  for (size_t i = 0; i < ctxt->candidate_set->size(); ++i) {
    CandidatePartition candidate = ctxt->candidate_set->at(i);
    if (candidate->sub_graph_->kind_ > kInjective) {
      continue;
    }
    for (PostDfsIndex exit_index : candidate->sub_graph_->exit_) {
      exit_map[exit_index].Add(i);
    }
  }

  // The two-step I -> tuple -> I rule.
  // Look all possible tuple consumers...
  for (size_t i = 0; i < ctxt->candidate_set->size(); ++i) {
    CandidatePartition tuple_consumer_candidate = ctxt->candidate_set->at(i);
    if (tuple_consumer_candidate->sub_graph_->kind_ > kInjective) {
      continue;
    }
    // For all possible tuples feeding into candidate...
    for (PostDfsIndex input_index : tuple_consumer_candidate->sub_graph_->input_) {
      auto node = ctxt->dataflow_graph->index_to_node(input_index);
      Expr sub_expr = node->ref();
      const auto* tuple_node = sub_expr.as<TupleNode>();
      if (tuple_node == nullptr) {
        continue;
      }
      // The tuple_consumer_candidate candidate consumes (at least one) tuple, eg as an argument
      // to an operator.
      // eg: concatenate((field1, ..., fieldn))
      auto tuple_dataflow_node = ctxt->dataflow_graph->item_to_node(tuple_node);

      // Collect all the possible unions. There may be more than one if different candidates
      // could supply the same tuple field.
      std::vector<std::vector<CandidatePartition>> all_possible_unions;

      // Obviously we must include the consumer.
      all_possible_unions.emplace_back();
      all_possible_unions.back().emplace_back(tuple_consumer_candidate);

      // We must include the tuple itself.
      SubGraph tuple_sub_graph(*ctxt->dataflow_graph,
                               IndexSet(ctxt->dataflow_graph->size(), {node->index_}), kInjective,
                               "tuple");
      CandidatePartition tuple_candidate("", std::move(tuple_sub_graph),
                                         tuple_consumer_candidate->partition_spec());
      all_possible_unions.back().emplace_back(std::move(tuple_candidate));

      // For all tuple fields...
      bool all_tuple_fields_have_producer = true;
      for (auto* tuple_field_dataflow_node : tuple_dataflow_node->inputs_) {
        // Collect all the candidates which could produce this tuple field.
        std::vector<CandidatePartition> to_appends;
        size_t start_j =
            i < ctxt->candidate_set->first_new_index() ? ctxt->candidate_set->first_new_index() : 0;
        for (size_t j : exit_map[tuple_field_dataflow_node->index_]) {
          if (i == j) {
            continue;
          }
          if (i < start_j) {
            // We already explored the cross-product of candidates [0, first_new_index), so don't
            // do it again.
            continue;
          }
          CandidatePartition tuple_field_producer = ctxt->candidate_set->at(j);
          // The tuple_field_producer candidate can provide this tuple field.
          // eg concatenate((..., producer, ...))
          to_appends.emplace_back(tuple_field_producer);
        }
        if (to_appends.empty()) {
          // At least one of the tuple's fields does not have a producer candidate we can
          // union in, so we need to give up.
          all_tuple_fields_have_producer = false;
          break;
        } else {
          // If to_appends = [A, B] and we already have possible unions [C, D] and [E, F] then
          // the new possible unions are [C, D, A], [C, D, B], [E, F, A] and [E, F, B].
          std::vector<std::vector<CandidatePartition>> new_all_possible_unions;
          for (const auto& to_append : to_appends) {
            for (const auto& possible_union : all_possible_unions) {
              new_all_possible_unions.emplace_back(possible_union);
              new_all_possible_unions.back().emplace_back(to_append);
            }
          }
          all_possible_unions = std::move(new_all_possible_unions);
        }
      }

      if (!all_tuple_fields_have_producer) {
        continue;
      }

      // Actually build the candidates which union according to all_possible_unions.
      for (const auto& possible_union : all_possible_unions) {
        if (possible_union.size() > 2) {
          CandidatePartition new_candidate =
              CandidatePartition::DisjointUnion(*ctxt->dataflow_graph, possible_union);
#if TVM_LOG_DEBUG
          std::ostringstream os;
          bool first = true;
          for (const auto& candidate : possible_union) {
            if (first) {
              first = false;
            } else {
              os << ", ";
            }
            os << candidate->ToString();
          }
          VLOG(2) << "Fired rule " << rule_name_ << " on {" << os.str() << "} to yield "
                  << new_candidate->ToString();
#endif
          ctxt->candidate_set->Add(*ctxt->dataflow_graph, new_candidate);
        }
      }
    }
  }
}

std::string TupleArgCombinerRuleNode::ToString() const {
  return "TupleArgCombinerRule(" + rule_name_ + ")";
}

TupleArgCombinerRule::TupleArgCombinerRule(String rule_name) {
  auto node = runtime::make_object<TupleArgCombinerRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(TupleProjCombinerRuleNode);

void TupleProjCombinerRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

void TupleProjCombinerRuleNode::AppendAllResults(AppendAllResultsContext* ctxt) const {
  VLOG(1) << "running TupleProjCombinerRule(" << rule_name_ << ")";
  // We already explored [0, first_new_index), so don't do it again.
  for (size_t i = ctxt->candidate_set->first_new_index(); i < ctxt->candidate_set->size(); ++i) {
    CandidatePartition base = ctxt->candidate_set->at(i);
    for (PostDfsIndex index : base->sub_graph_->output_) {
      auto node = ctxt->dataflow_graph->index_to_node(index);
      if (node->ref().as<TupleGetItemNode>()) {
        IndexSet index_set(ctxt->dataflow_graph->size(), {node->index_});
        SubGraph sub_graph(*ctxt->dataflow_graph, std::move(index_set), kInjective, "proj");
        CandidatePartition proj_candidate("", std::move(sub_graph), base->spec_);
        CandidatePartition new_candidate =
            base.DisjointUnion(*ctxt->dataflow_graph, proj_candidate);
        VLOG(2) << "Fired rule " << rule_name_ << " on " << proj_candidate->ToString() << " and "
                << base->ToString() << " to yield " << new_candidate->ToString();
        ctxt->candidate_set->Add(*ctxt->dataflow_graph, new_candidate);
      }
    }
  }
}

std::string TupleProjCombinerRuleNode::ToString() const {
  return "TupleProjCombinerRule(" + rule_name_ + ")";
}

TupleProjCombinerRule::TupleProjCombinerRule(String rule_name) {
  auto node = runtime::make_object<TupleProjCombinerRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(ConstantCombinerRuleNode);

void ConstantCombinerRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

void ConstantCombinerRuleNode::AppendAllResults(AppendAllResultsContext* ctxt) const {
  VLOG(1) << "running ConstantCombinerRule(" << rule_name_ << ")";
  // We already explored [0, first_new_index), so don't do it again.
  for (size_t i = ctxt->candidate_set->first_new_index(); i < ctxt->candidate_set->size(); ++i) {
    CandidatePartition base = ctxt->candidate_set->at(i);
    IndexSet new_constants(ctxt->dataflow_graph->size());
    for (PostDfsIndex index : base->sub_graph_->input_) {
      auto node = ctxt->dataflow_graph->index_to_node(index);
      if (node->ref().as<ConstantNode>()) {
        new_constants.Add(index);
      }
    }
    if (!new_constants.IsZero()) {
      SubGraph sub_graph(*ctxt->dataflow_graph, new_constants, kElemWise, "const");
      CandidatePartition new_const_candidate("", std::move(sub_graph), base->spec_);
      CandidatePartition new_candidate =
          base.DisjointUnion(*ctxt->dataflow_graph, new_const_candidate);
      VLOG(2) << "Fired rule " << rule_name_ << " on " << new_const_candidate->ToString() << " and "
              << base->ToString() << " to yield " << new_candidate->ToString();
      ctxt->candidate_set->Add(*ctxt->dataflow_graph, new_candidate);
    }
  }
}

std::string ConstantCombinerRuleNode::ToString() const {
  return "ConstantCombinerRule(" + rule_name_ + ")";
}

ConstantCombinerRule::ConstantCombinerRule(String rule_name) {
  auto node = runtime::make_object<ConstantCombinerRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
