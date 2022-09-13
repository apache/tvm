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
 * \file src/relay/collage/partition_rule.cc
 * \brief Compositional partitioning rules.
 */

#include "./partition_rule.h"

#include <tvm/relay/transform.h>

#include "./partition_rule.h"
#include "./partition_spec.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

TVM_REGISTER_NODE_TYPE(PartitionRuleNode);

void PartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> PartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  ICHECK(false) << "PartitionRuleNode::AllCandidates should be overridden in sub-class";
  return {};
}

std::string PartitionRuleNode::ToString() const { return ToDoc().str(); }

Doc PartitionRuleNode::ToDoc() const {
  Doc doc;
  doc << GetTypeKey() << "(" << Doc::NewLine(2);
  std::vector<Doc> body_items;
  AppendBodyItems(&body_items);
  doc << Doc::Indent(2, Doc::Concat(body_items, Doc::NewLine())) << Doc::NewLine();
  doc << ")";
  return doc;
}

void PartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  body_items->emplace_back();
  body_items->back() << "rule_name=" << Doc::StrLiteral(rule_name_);
}

PartitionRule::PartitionRule(String rule_name) {
  auto node = runtime::make_object<PartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

bool DefaultPatternPredicate(const Expr& matched_sub_expr) { return true; }

TVM_REGISTER_NODE_TYPE(DFPatternPartitionRuleNode);

void DFPatternPartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> DFPatternPartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  VLOG(1) << "running DFPatternPartitionRule(" << rule_name_ << ")";
  std::vector<CandidatePartition> result;
  DFPatternMatcher matcher(&dataflow_graph.indexed_graph());
  for (PostDfsIndex index = 0; index < dataflow_graph.size(); ++index) {
    Expr sub_expr = dataflow_graph.index_to_node(index)->ref();
    if (!matcher.Match(pattern_, sub_expr)) {
      continue;
    }
    if (!predicate_(sub_expr)) {
      VLOG(1) << "DFPatternPartitionRule(" << rule_name_ << ") has failing predicate";
      continue;
    }
    IndexSet inside = MatcherToIndexSet(matcher);
    auto [kind, label] = SubGraphKindAndLabel(dataflow_graph, inside);
    SubGraph sub_graph(dataflow_graph, std::move(inside), kind, std::move(label));
    String rule_name = rule_name_.empty() ? sub_graph->label_ : rule_name_;
    CandidatePartition candidate(std::move(rule_name), std::move(sub_graph), spec);
    VLOG(2) << "DFPatternPartitionRule(" << rule_name_ << ") yields " << candidate->ToString();
    result.emplace_back(std::move(candidate));
  }
  VLOG(1) << "DFPatternPartitionRule(" << rule_name_ << ") produced " << result.size()
          << " candidates";
  return result;
}

void DFPatternPartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  PartitionRuleNode::AppendBodyItems(body_items);
  body_items->emplace_back();
  body_items->back() << "pattern=" << PrettyPrint(pattern_);
}

DFPatternPartitionRule::DFPatternPartitionRule(String rule_name, DFPattern pattern,
                                               TPatternPredicate predicate) {
  auto node = runtime::make_object<DFPatternPartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->pattern_ = std::move(pattern);
  node->predicate_ = std::move(predicate);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(CompositePartitionRuleNode);

void CompositePartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> CompositePartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  std::vector<CandidatePartition> candidates = sub_rule_->AllCandidates(dataflow_graph, spec);
  VLOG(1) << "running CompositePartitionRule(" << rule_name_ << ") over " << candidates.size()
          << " sub-candidates";
  std::vector<CandidatePartition> result;
  FunctionAttrsMap attrs;
  attrs.Set(attr::kComposite, rule_name_);
  for (auto& candidate : candidates) {
    String rule_name = NestLabels(rule_name_, candidate->rule_name_);
    SubGraph sub_graph = candidate->sub_graph_.WithAttrs(dataflow_graph, attrs);
    CandidatePartition new_candidate = WithSubGraph(
        WithRuleName(std::move(candidate), std::move(rule_name)), std::move(sub_graph));
    VLOG(2) << "CompositePartitionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
    result.emplace_back(std::move(new_candidate));
  }
  VLOG(1) << "CompositePartitionRule(" << rule_name_ << ") produced " << result.size()
          << " candidates";
  return result;
}

void CompositePartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  PartitionRuleNode::AppendBodyItems(body_items);
  body_items->emplace_back();
  body_items->back() << "sub_rule=" << sub_rule_->ToDoc();
}

CompositePartitionRule::CompositePartitionRule(String rule_name, PartitionRule sub_rule) {
  auto node = runtime::make_object<CompositePartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(PrimitivePartitionRuleNode);

void PrimitivePartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> PrimitivePartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  std::vector<CandidatePartition> candidates = sub_rule_->AllCandidates(dataflow_graph, spec);
  VLOG(1) << "running PrimitivePartitionRule(" << rule_name_ << ") over " << candidates.size()
          << " sub-candidates";
  std::vector<CandidatePartition> result;
  FunctionAttrsMap attrs;
  attrs.Set(attr::kPrimitive, Integer(1));
  if (spec->target_.IsExternalCodegen()) {
    // The spec name will be the target kind name which is 1:1 with the "Compiler" attribute name.
    attrs.Set(attr::kCompiler, spec->spec_name_);
  }
  for (auto& candidate : candidates) {
    String rule_name = NestLabels(rule_name_, candidate->rule_name_);
    SubGraph sub_graph = candidate->sub_graph_.WithAttrs(dataflow_graph, attrs);
    CandidatePartition new_candidate = WithSubGraph(
        WithRuleName(std::move(candidate), std::move(rule_name)), std::move(sub_graph));
    VLOG(2) << "PrimitivePartitionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
    result.emplace_back(std::move(new_candidate));
  }
  VLOG(1) << "PrimitivePartitionRule(" << rule_name_ << ") produced " << result.size()
          << " candidates";
  return result;
}

void PrimitivePartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  PartitionRuleNode::AppendBodyItems(body_items);
  body_items->emplace_back();
  body_items->back() << "sub_rule=" << sub_rule_->ToDoc();
}

PrimitivePartitionRule::PrimitivePartitionRule(String rule_name, PartitionRule sub_rule) {
  auto node = runtime::make_object<PrimitivePartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(UnionPartitionRuleNode);

void UnionPartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> UnionPartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  std::vector<CandidatePartition> result;
  for (const auto& sub_rule : sub_rules_) {
    std::vector<CandidatePartition> candidates = sub_rule->AllCandidates(dataflow_graph, spec);
    for (auto& candidate : candidates) {
      String rule_name = NestLabels(rule_name_, candidate->rule_name_);
      CandidatePartition new_candidate = WithRuleName(std::move(candidate), std::move(rule_name));
      VLOG(2) << "UnionPartitionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
      result.emplace_back(std::move(new_candidate));
    }
  }
  VLOG(1) << "UnionPartitionRule(" << rule_name_ << ") produced " << result.size() << " candidates";
  return result;
}

void UnionPartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  PartitionRuleNode::AppendBodyItems(body_items);
  for (const auto& sub_rule : sub_rules_) {
    body_items->emplace_back();
    body_items->back() << "sub_rule=" << sub_rule->ToDoc();
  }
}

UnionPartitionRule::UnionPartitionRule(String rule_name, Array<PartitionRule> sub_rules) {
  auto node = runtime::make_object<UnionPartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rules_ = std::move(sub_rules);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(OpCallByKindPartitionRuleNode);

void OpCallByKindPartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> OpCallByKindPartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  VLOG(1) << "running OpCallByKindPartitionRule(" << rule_name_ << ")";
  std::vector<CandidatePartition> result;
  for (PostDfsIndex index = 0; index < dataflow_graph.size(); ++index) {
    auto node = dataflow_graph.index_to_node(index);
    Expr sub_expr = node->ref();
    if (sub_expr->IsInstance<CallNode>()) {
      auto [kind, label] = SubExprKindAndLabel(sub_expr);
      if (kind <= kOutEWiseFusable) {
        IndexSet inside(dataflow_graph.size(), {index});
        SubGraph sub_graph(dataflow_graph, std::move(inside), kind, std::move(label));
        String rule_name = NestLabels(rule_name_, sub_graph->label_);
        CandidatePartition candidate(std::move(rule_name), std::move(sub_graph), spec);
        VLOG(2) << "OpCallByKindPartitionRule(" << rule_name_ << ") yields "
                << candidate->ToString();
        result.emplace_back(std::move(candidate));
      }
    }
  }
  VLOG(1) << "OpCallByKindPartitionRule(" << rule_name_ << ") produced " << result.size()
          << " candidates";
  return result;
}

void OpCallByKindPartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  PartitionRuleNode::AppendBodyItems(body_items);
}

OpCallByKindPartitionRule::OpCallByKindPartitionRule(String rule_name) {
  auto node = runtime::make_object<OpCallByKindPartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(CombinePartitionRuleNode);

void CombinePartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> CombinePartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  // We'll accumulate all the candidates here, starting with those from the sub-rule.
  // Once a candidate is added to this vector it is immutable.
  std::vector<CandidatePartition> candidates = sub_rule_->AllCandidates(dataflow_graph, spec);
  VLOG(1) << "running CombinePartitionRule(" << rule_name_ << ") over " << candidates.size()
          << " sub-candidates";
  CandidateSet result_set(std::move(candidates));

  size_t num_rounds = 0;
  AppendAllResultsContext ctxt(&dataflow_graph, max_depth_, &result_set);
  while (result_set.PrepareForNextRound()) {
    VLOG_CONTEXT << "round " << ++num_rounds;
    VLOG(1) << "checking " << result_set.size() << " candidates (" << result_set.first_new_index()
            << " existing)";
    for (const auto& combiner_rule : combiner_rules_) {
      combiner_rule->AppendAllResults(&ctxt);
    }
  }

  std::vector<CandidatePartition> result;
  for (auto& candidate : result_set.MovedCurrentCandidates()) {
    String rule_name = NestLabels(rule_name_, candidate->rule_name_);
    CandidatePartition new_candidate = WithRuleName(std::move(candidate), std::move(rule_name));
    VLOG(2) << "CombinePartitionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
    result.emplace_back(std::move(new_candidate));
  }
  VLOG(1) << "CombinePartitionRule(" << rule_name_ << ") produced " << result.size()
          << " candidates";
  return result;
}

void CombinePartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  PartitionRuleNode::AppendBodyItems(body_items);
  body_items->emplace_back();
  body_items->back() << "sub_rule=" << sub_rule_->ToDoc();
  for (const auto& combiner_rule : combiner_rules_) {
    body_items->emplace_back();
    body_items->back() << "combiner_rule=" << combiner_rule->ToString();
  }
  body_items->emplace_back();
  body_items->back() << "max_depth=" << max_depth_;
}

CombinePartitionRule::CombinePartitionRule(String rule_name, PartitionRule sub_rule,
                                           Array<CombinerRule> combiner_rules, size_t max_depth_) {
  auto node = runtime::make_object<CombinePartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  node->combiner_rules_ = std::move(combiner_rules);
  node->max_depth_ = max_depth_;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(OnlyValidPartitionRuleNode);

void OnlyValidPartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> OnlyValidPartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  std::vector<CandidatePartition> candidates = sub_rule_->AllCandidates(dataflow_graph, spec);
  VLOG(1) << "running OnlyValidPartitionRule(" << rule_name_ << ") over " << candidates.size()
          << " sub-candidates";
  std::vector<CandidatePartition> result;
  for (auto& candidate : candidates) {
    if (!candidate->sub_graph_->IsValid(dataflow_graph, config_)) {
      VLOG(2) << "Ignoring invalid candidate " << candidate->ToString();
      continue;
    }
    String rule_name = NestLabels(rule_name_, candidate->rule_name_);
    CandidatePartition new_candidate = WithRuleName(std::move(candidate), std::move(rule_name));
    VLOG(2) << "OnlyValidPartitionRule(" << rule_name_ << ") yields " << new_candidate->ToString();
    result.emplace_back(std::move(new_candidate));
  }
  VLOG(1) << "OnlyValidPartitionRule(" << rule_name_ << ") produced " << result.size()
          << " candidates";
  return result;
}

void OnlyValidPartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {
  PartitionRuleNode::AppendBodyItems(body_items);
  body_items->emplace_back();
  body_items->back() << "sub_rule=" << sub_rule_->ToDoc();
  body_items->emplace_back();
  body_items->back() << "config=" << config_.ToString();
}

OnlyValidPartitionRule::OnlyValidPartitionRule(String rule_name, PartitionRule sub_rule,
                                               const SubGraphConfig& config) {
  auto node = runtime::make_object<OnlyValidPartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  node->sub_rule_ = std::move(sub_rule);
  node->config_ = config;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(HostPartitionRuleNode);

void HostPartitionRuleNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> HostPartitionRuleNode::AllCandidates(
    const DataflowGraph& dataflow_graph, const PartitionSpec& spec) const {
  VLOG(1) << "running HostPartitionRule(" << rule_name_ << ")";
  std::vector<CandidatePartition> result;
  for (PostDfsIndex index = 0; index < dataflow_graph.size(); ++index) {
    if (MustBeLowered(dataflow_graph.index_to_node(index)->ref())) {
      continue;
    }
    IndexSet inside(dataflow_graph.size(), {index});
    auto [kind, label] = SubGraphKindAndLabel(dataflow_graph, inside);
    SubGraph sub_graph(dataflow_graph, std::move(inside), kind, label);
    String rule_name = NestLabels(rule_name_, sub_graph->label_);
    // We'll a zero cost for the candidate since we'll never want to actually estimate the cost
    // of this 'partition'.
    CandidatePartition candidate(std::move(rule_name), std::move(sub_graph), spec, Cost::Zero());
    VLOG(2) << "HostPartitionRule(" << rule_name_ << ") yields " << candidate->ToString();
    result.push_back(candidate);
  }
  VLOG(1) << "HostPartitionRule(" << rule_name_ << ") produced " << result.size() << " candidates";
  return result;
}

void HostPartitionRuleNode::AppendBodyItems(std::vector<Doc>* body_items) const {}

HostPartitionRule::HostPartitionRule(String rule_name) {
  auto node = runtime::make_object<HostPartitionRuleNode>();
  node->rule_name_ = std::move(rule_name);
  data_ = std::move(node);
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
