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
 * \file src/tvm/relay/dataflow_matcher.h
 * \brief The dataflow pattern matcher for Relay.
 */
#ifndef TVM_RELAY_IR_DATAFLOW_MATCHER_H_
#define TVM_RELAY_IR_DATAFLOW_MATCHER_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <vector>

#include "indexed_graph.h"

namespace tvm {
namespace relay {

// Pattern Matcher

class DominatorMatcher;

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  explicit DFPatternMatcher(const Expr& root_expr) : expr_graph_(CreateIndexedGraph(root_expr)) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Array<Expr>> GetMemo() { return Map<DFPattern, Array<Expr>>(memo_); }
  const IndexedGraph<Expr> expr_graph_;

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const AltPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const IfPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const LetPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  bool MatchesPath(const DominatorPatternNode* op, const Expr& expr);
  bool DominatesParent(const DominatorPatternNode* op, const Expr& expr);

  std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
  bool memoize_ = true;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_IR_DATAFLOW_MATCHER_H_
