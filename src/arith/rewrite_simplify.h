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
 * \file rewrite_simplify.h
 * \brief Rewrite-rule based simplification.
 */
#ifndef TVM_ARITH_REWRITE_SIMPLIFY_H_
#define TVM_ARITH_REWRITE_SIMPLIFY_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <unordered_map>
#include <vector>
#include "const_fold.h"
#include "pattern_match.h"
#include "ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace tir;

/*!
 * \brief Rewrite-based simplifier.
 *
 * This class can be inheritated for other simplifiers.
 */
class RewriteSimplifier::Impl : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::VisitExpr_;

  explicit Impl(Analyzer* parent)
      : IRMutatorWithAnalyzer(parent) {}

  void Update(const Var& var, const PrimExpr& info, bool override_info);
  PrimExpr VisitExpr_(const AddNode* op) override;
  PrimExpr VisitExpr_(const SubNode* op) override;
  PrimExpr VisitExpr_(const MulNode* op) override;
  PrimExpr VisitExpr_(const DivNode* op) override;
  PrimExpr VisitExpr_(const ModNode* op) override;
  PrimExpr VisitExpr_(const FloorDivNode* op) override;
  PrimExpr VisitExpr_(const FloorModNode* op) override;
  PrimExpr VisitExpr_(const MinNode* op) override;
  PrimExpr VisitExpr_(const MaxNode* op) override;
  PrimExpr VisitExpr_(const EQNode* op) override;
  PrimExpr VisitExpr_(const NENode* op) override;
  PrimExpr VisitExpr_(const LTNode* op) override;
  PrimExpr VisitExpr_(const LENode* op) override;
  PrimExpr VisitExpr_(const GTNode* op) override;
  PrimExpr VisitExpr_(const GENode* op) override;
  PrimExpr VisitExpr_(const AndNode* op) override;
  PrimExpr VisitExpr_(const OrNode* op) override;
  PrimExpr VisitExpr_(const NotNode* op) override;
  PrimExpr VisitExpr_(const SelectNode* op) override;
  PrimExpr VisitExpr_(const CallNode* op) override;
  PrimExpr VisitExpr_(const VarNode* op) override;
  PrimExpr VisitExpr_(const CastNode* op) override;
  PrimExpr VisitExpr_(const LetNode* op) override;

  std::function<void()> EnterConstraint(const PrimExpr& constraint);

 protected:
  /*! \brief internal structure for comparison. */
  enum CompareResult {
    kUnknown,
    kEQ,
    kGT,
    kGE,
    kLT,
    kLE,
    kNE
  };
  // counter to record recursive rewrite depth.
  int recur_depth_{0};
  // internal variable map
  std::unordered_map<Var, PrimExpr, ObjectHash, ObjectEqual> var_map_;

  std::vector<PrimExpr> literal_constraints_;

  // maximum number of recursion allowed during a single pass.
  static const constexpr int kMaxRecurDepth = 5;

  /*!
   * \brief try to compare x against val.
   * \param x The expression to be evaluated.
   * \param val The constant value.
   * \return comparison result.
   */
  CompareResult TryCompare(const PrimExpr& x, int64_t val);

 private:
  // Whether x >= val
  bool CanProveGreaterEqual(const PrimExpr& x, int64_t val) {
    return analyzer_->CanProveGreaterEqual(x, val);
  }
  // Whether x == val
  bool CanProveEqual(const PrimExpr& x, int64_t val) {
    // TODO(tqchen) refer back to super-analyzer.
    return TryCompare(x, val) == kEQ;
  }

  // Recursive rewrite x
  // we limit maximum depth of recursive rewrite allowed to
  // avoid infinite loop
  PrimExpr RecursiveRewrite(const PrimExpr& x) {
    if (recur_depth_ >= kMaxRecurDepth) return x;
    ++recur_depth_;
    PrimExpr res = this->VisitExpr(x);
    --recur_depth_;
    return res;
  }

  template<typename TA>
  PConstWithTypeLike<TA> ZeroWithTypeLike(const Pattern<TA>& pattern) {
    return PConstWithTypeLike<TA>(pattern.derived(), 0);
  }

  template<typename TA>
  PConstWithTypeLike<TA> OneWithTypeLike(const Pattern<TA>& pattern) {
    return PConstWithTypeLike<TA>(pattern.derived(), 1);
  }
};


}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_REWRITE_SIMPLIFY_H_
