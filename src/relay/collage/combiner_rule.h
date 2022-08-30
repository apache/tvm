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
 * \file src/relay/collage/combiner_rule.h
 * \brief Helpers for the \p CombinePartitionRule
 */

#ifndef TVM_RELAY_COLLAGE_COMBINER_RULE_H_
#define TVM_RELAY_COLLAGE_COMBINER_RULE_H_

#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/expr.h>

#include <string>

#include "./candidate_partition.h"
#include "./candidate_set.h"
#include "./sub_graph.h"

namespace tvm {
namespace relay {
namespace collage {

/*!
 * \brief Base class for all 'simple' combiner rules.
 *
 * Given \p upstream and \p downstream candidates which touch, a simple combiner rule returns
 * true if their union should also be considered a candidate.
 */
class SimpleCombinerRuleNode : public Object {
 public:
  String rule_name_;

  void VisitAttrs(AttrVisitor* v);

  virtual bool Fires(const DataflowGraph& dataflow_graph, const CandidatePartition& upstream,
                     const CandidatePartition& downstream) const;

  virtual std::string ToString() const;

  static constexpr const char* _type_key = "relay.collage.SimpleCombinerRule";
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_DECLARE_BASE_OBJECT_INFO(SimpleCombinerRuleNode, Object);
};

class SimpleCombinerRule : public ObjectRef {
 public:
  explicit SimpleCombinerRule(String rule_name);

  TVM_DEFINE_OBJECT_REF_METHODS(SimpleCombinerRule, ObjectRef, SimpleCombinerRuleNode);
};

/*!
 * \brief A simple combiner rule which fires if the \p upstream and \p downstream candidates have
 * the given \p upstream_kind and \p downstream_kind (or less) respectively.
 */
class ByKindSimpleCombinerRuleNode : public SimpleCombinerRuleNode {
 public:
  OpPatternKind upstream_kind_;
  OpPatternKind downstream_kind_;

  void VisitAttrs(AttrVisitor* v);

  bool Fires(const DataflowGraph& dataflow_graph, const CandidatePartition& upstream,
             const CandidatePartition& downstream) const override;
  std::string ToString() const override;

  static constexpr const char* _type_key = "relay.collage.ByKindSimpleCombinerRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ByKindSimpleCombinerRuleNode, SimpleCombinerRuleNode);
};

class ByKindSimpleCombinerRule : public SimpleCombinerRule {
 public:
  ByKindSimpleCombinerRule(OpPatternKind upstream_kind, OpPatternKind downstream_kind);

  TVM_DEFINE_OBJECT_REF_METHODS(ByKindSimpleCombinerRule, SimpleCombinerRule,
                                ByKindSimpleCombinerRuleNode);
};

/*! \brief Context required by CombineRuleNode::AppendAllResultsContext. */
struct AppendAllResultsContext {
  AppendAllResultsContext(const DataflowGraph* dataflow_graph, size_t max_depth,
                          CandidateSet* candidate_set)
      : dataflow_graph(dataflow_graph), max_depth(max_depth), candidate_set(candidate_set) {}

  const DataflowGraph* dataflow_graph;
  size_t max_depth;
  CandidateSet* candidate_set;
};

/*!
 * \brief Base class for all 'combiner' rules.
 *
 * Given the current candidate set, a combiner rule looks for opportunities to form larger
 * candidates, optionally removing existing candidates in the process.
 */
class CombinerRuleNode : public Object {
 public:
  String rule_name_;

  void VisitAttrs(AttrVisitor* v);

  virtual void AppendAllResults(AppendAllResultsContext* ctxt) const;
  virtual std::string ToString() const;

  static constexpr const char* _type_key = "relay.collage.CombinerRule";
  static constexpr const uint32_t _type_child_slots = 4;
  TVM_DECLARE_BASE_OBJECT_INFO(CombinerRuleNode, Object);
};

class CombinerRule : public ObjectRef {
 public:
  explicit CombinerRule(String rule_name);

  TVM_DEFINE_OBJECT_REF_METHODS(CombinerRule, ObjectRef, CombinerRuleNode);
};

/*!
 * \brief A combiner rule which runs one or more simple combiner rules over the current
 * touching candidates.
 */
class AllSimpleCombinerRuleNode : public CombinerRuleNode {
 public:
  Array<SimpleCombinerRule> simple_rules_;

  void VisitAttrs(AttrVisitor* v);

  void AppendAllResults(AppendAllResultsContext* ctxt) const override;
  std::string ToString() const override;

  static constexpr const char* _type_key = "relay.collage.AllSimpleCombinerRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllSimpleCombinerRuleNode, CombinerRuleNode);
};

class AllSimpleCombinerRule : public CombinerRule {
 public:
  AllSimpleCombinerRule(String rule_name, Array<SimpleCombinerRule> simple_rules);

  TVM_DEFINE_OBJECT_REF_METHODS(AllSimpleCombinerRule, CombinerRule, AllSimpleCombinerRuleNode);
};

/*!
 * \brief A combiner rule which combines injective sub-groups which appear inside tuples which are
 * themselves inputs to injective sub-groups.
 */
class TupleArgCombinerRuleNode : public CombinerRuleNode {
 public:
  void VisitAttrs(AttrVisitor* v);

  void AppendAllResults(AppendAllResultsContext* ctxt) const override;
  std::string ToString() const override;

  static constexpr const char* _type_key = "relay.collage.TupleArgCombinerRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleArgCombinerRuleNode, CombinerRuleNode);
};

class TupleArgCombinerRule : public CombinerRule {
 public:
  explicit TupleArgCombinerRule(String rule_name);

  TVM_DEFINE_OBJECT_REF_METHODS(TupleArgCombinerRule, CombinerRule, TupleArgCombinerRuleNode);
};

/*!
 * \brief A combiner rule which combines tuple projection if it's an output of an injective
 * group.
 */
class TupleProjCombinerRuleNode : public CombinerRuleNode {
 public:
  void VisitAttrs(AttrVisitor* v);

  void AppendAllResults(AppendAllResultsContext* ctxt) const override;
  std::string ToString() const override;

  static constexpr const char* _type_key = "relay.collage.TupleProjCombinerRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleProjCombinerRuleNode, CombinerRuleNode);
};

class TupleProjCombinerRule : public CombinerRule {
 public:
  explicit TupleProjCombinerRule(String rule_name);

  TVM_DEFINE_OBJECT_REF_METHODS(TupleProjCombinerRule, CombinerRule, TupleProjCombinerRuleNode);
};

/*!
 * \brief A combiner rule which combines constants in argument positions to existing candidates.
 * Note that scalars are always inlined, so this rule only combines tensor constant arguments.
 */
class ConstantCombinerRuleNode : public CombinerRuleNode {
 public:
  void VisitAttrs(AttrVisitor* v);

  void AppendAllResults(AppendAllResultsContext* ctxt) const override;
  std::string ToString() const override;

  static constexpr const char* _type_key = "relay.collage.ConstantCombinerRule";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantCombinerRuleNode, CombinerRuleNode);
};

class ConstantCombinerRule : public CombinerRule {
 public:
  explicit ConstantCombinerRule(String rule_name);

  TVM_DEFINE_OBJECT_REF_METHODS(ConstantCombinerRule, CombinerRule, ConstantCombinerRuleNode);
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_COMBINER_RULE_H_
