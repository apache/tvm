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
 * \file src/tvm/relax/dataflow_matcher.h
 * \brief The auxiliary data structure for dataflow matcher.
 */
#ifndef TVM_RELAX_IR_DATAFLOW_MATCHER_H_
#define TVM_RELAX_IR_DATAFLOW_MATCHER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/dataflow_pattern_functor.h>

#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  using var2val_t = runtime::Map<Var, Expr>;

  explicit DFPatternMatcher() {}
  explicit DFPatternMatcher(var2val_t var2val) : var2val_(std::move(var2val)) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Expr> GetMemo() { return memo_; }

  /* \brief Unwrap trivial expressions/bindings */
  static Expr UnwrapBindings(Expr expr, const Map<Var, Expr>& bindings);

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const OrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AndPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const NotPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const StructInfoPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;

  bool VisitDFPattern_(const DataflowVarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const GlobalVarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExternFuncPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const PrimArrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const UnorderedTuplePatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  bool TryUnorderedMatch(size_t idx, const tvm::Array<DFPattern> patterns,
                         const tvm::Array<Expr> fields, std::vector<int8_t>& match_cache,
                         std::vector<bool>& matched);

  /* \brief Simplify a boolean condition using the analyzer
   *
   * Matching struct info can often produce conditions that do not
   * simplify cleanly.  For example, while the rewrite simplifier can
   * recognize that `m==0 && m==1` can be simplifies to `false`, it
   * cannot recognize that `m==0 && n==0 && m==1` can be simplified to
   * false.
   *
   * This function applies additional simplification steps to handle
   * these cases, before delgating to `analyzer_.Simplify`.
   */
  PrimExpr SimplifyCondition(PrimExpr condition);

  std::unordered_map<DFPattern, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
  var2val_t var2val_;
  std::vector<DFPattern> matched_nodes_;
  PrimExpr symbolic_expr_condition_{Bool(true)};
  arith::Analyzer analyzer_;
  bool memoize_ = true;
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_DATAFLOW_MATCHER_H_
