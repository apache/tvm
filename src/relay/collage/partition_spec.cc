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
 * \file src/relay/collage/partition_spec.cc
 * \brief Combine a \p PartitionRule with a \p Target.
 */

#include "./partition_spec.h"

#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {

String DefaultValidateSubGraphFunc(const Function& function) { return String(); }

TVM_REGISTER_NODE_TYPE(PartitionSpecNode);

void PartitionSpecNode::VisitAttrs(AttrVisitor* v) {
  // TODO(mbs)
}

std::vector<CandidatePartition> PartitionSpecNode::AllCandidates(
    const DataflowGraph& dataflow_graph) const {
  std::vector<CandidatePartition> result;
  // Make sure the target is in scope for inspection by any predicates in
  // DFPatternPartitionRuleNode rules.
  With<Target> target_scope(target_);
  // Gather all the candidates.
  std::vector<CandidatePartition> candidates =
      rule_->AllCandidates(dataflow_graph, GetRef<PartitionSpec>(this));
  // Update the rules names.
  for (const auto& candidate : candidates) {
    ICHECK_EQ(candidate->spec_, GetRef<PartitionSpec>(this));
    String rule_name = NestLabels(spec_name_, candidate->rule_name_);
    CandidatePartition new_candidate = WithRuleName(candidate, std::move(rule_name));
    result.emplace_back(std::move(new_candidate));
  }
  return result;
}

std::string PartitionSpecNode::ToString() const {
  Doc doc;
  doc << "PartitionSpec(" << Doc::NewLine(2);
  std::vector<Doc> body_items;
  body_items.emplace_back();
  body_items.back() << "spec_name=" << Doc::StrLiteral(spec_name_);
  body_items.emplace_back();
  body_items.back() << "target=" << target_->ToDebugString();
  body_items.emplace_back();
  body_items.back() << "rule=" << rule_->ToDoc();
  doc << Doc::Indent(2, Doc::Concat(body_items, Doc::NewLine())) << Doc::NewLine();
  doc << ")";
  return doc.str();
}

PartitionSpec::PartitionSpec(String spec_name, Target target, PartitionRule rule,
                             TValidateSubGraphFunc validate_sub_graph_func) {
  auto node = runtime::make_object<PartitionSpecNode>();
  node->spec_name_ = std::move(spec_name);
  node->target_ = std::move(target);
  node->rule_ = std::move(rule);
  node->validate_sub_graph_func_ = std::move(validate_sub_graph_func);
  data_ = std::move(node);
}

}  // namespace collage
}  // namespace relay
}  // namespace tvm
