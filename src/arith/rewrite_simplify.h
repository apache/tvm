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

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "const_fold.h"
#include "ir_mutator_with_analyzer.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

/* \brief Usage counters for RewriteSimplifier
 *
 * These are intended for debug and testing purposes, to ensure that
 * PrimExpr simplifications and TIR passes do not require an excessive
 */
struct RewriteSimplifierStatsNode : Object {
  int64_t nodes_visited{0};
  int64_t constraints_entered{0};
  int64_t rewrites_attempted{0};
  int64_t rewrites_performed{0};
  int64_t max_recursive_depth{0};
  int64_t num_recursive_rewrites{0};

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("nodes_visited", &nodes_visited);
    v->Visit("constraints_entered", &constraints_entered);
    v->Visit("rewrites_attempted", &rewrites_attempted);
    v->Visit("rewrites_performed", &rewrites_performed);
    v->Visit("max_recursive_depth", &max_recursive_depth);
    v->Visit("num_recursive_rewrites", &num_recursive_rewrites);
  }

  static constexpr const char* _type_key = "arith.RewriteSimplifierStats";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteSimplifierStatsNode, Object);
};

struct RewriteSimplifierStats : ObjectRef {
  explicit RewriteSimplifierStats(RewriteSimplifierStatsNode data) {
    data_ = make_object<RewriteSimplifierStatsNode>(data);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(RewriteSimplifierStats, ObjectRef, RewriteSimplifierStatsNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RewriteSimplifierStatsNode);
};

/*!
 * \brief Rewrite-based simplifier.
 *
 * This class can be inheritated for other simplifiers.
 */
class RewriteSimplifier::Impl : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::VisitExpr_;

  explicit Impl(Analyzer* parent) : IRMutatorWithAnalyzer(parent) {}

  PrimExpr VisitExpr(const PrimExpr& e) override;

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

  /*! \brief Enable an optional extension or extensions
   *
   * \param flags A bitwise OR of all optional extensions that should
   * be enabled.
   */
  void SetEnabledExtensions(Extension flags);

  /*! \brief Return the currently enabled extensions */
  Extension GetEnabledExtensions() const;

  RewriteSimplifierStats GetStatsCounters() const { return RewriteSimplifierStats(stats_); }

  void ResetStatsCounters() { stats_ = {}; }

  void SetMaximumRewriteSteps(int64_t maximum) { maximum_rewrite_steps_ = maximum; }

 protected:
  int64_t maximum_rewrite_steps_{0};
  RewriteSimplifierStatsNode stats_;

  void RecordAttemptedRewrite() { stats_.rewrites_attempted++; }
  void RecordRewrite() {
    stats_.rewrites_performed++;

    ICHECK(maximum_rewrite_steps_ <= 0 || stats_.rewrites_performed <= maximum_rewrite_steps_)
        << "RewriteSimplifier exceeded maximum number of rewrites allowed ("
        << maximum_rewrite_steps_ << ")";
  }

  // counter to record recursive rewrite depth.
  int64_t recur_depth_{0};
  // internal variable map
  std::unordered_map<Var, PrimExpr> var_map_;

  std::vector<PrimExpr> literal_constraints_;

  // Optionally enabled extensions
  Extension enabled_extensions_{kNone};

  /*! Whether the simplifier is current
   */
  bool recursively_visiting_boolean_{false};

  // maximum number of recursion allowed during a single pass.
  static const constexpr int64_t kMaxRecurDepth = 5;
  /*!
   * \brief try to compare x against val.
   * \param x The expression to be evaluated.
   * \param val The constant value.
   * \return comparison result.
   */
  CompareResult TryCompare(const PrimExpr& x, int64_t val);

  /*! Try to compare x against y
   *
   * \param x The lhs of the comparison
   * \param y The rhs of the comparison
   * \return comparison result.
   */
  CompareResult TryCompare(const PrimExpr& x, const PrimExpr& y);

  /*!
   * \brief Internal function to check whether or not to inline let.
   * \param op The let expr.
   * \return The inline decision.
   */
  bool CanInlineLet(const LetNode* op);

  /*! \brief Internal function to apply constraints
   *
   * Tests whether the expression is known to be true or false based
   * on existing constraints.  If the expression or its negation
   * matches a constraint, return the boolean it should be replaced
   * with.  Otherwise, return false.
   */
  Optional<PrimExpr> TryMatchLiteralConstraint(const PrimExpr& expr) const;

  /*! \brief Rewrite rules for Less Than comparisons
   *
   * These are separate from the VisitExpr_(const LTNode*) method, as
   * they may required from rewrites of LT or LE.
   */
  PrimExpr ApplyRewriteRules(LT node);

  /*! \brief Rewrite rules for Equal comparisons
   *
   * These are separate from the VisitExpr_(const EQNode*) method, as
   * they may required from rewrites of LE or NE.
   */
  PrimExpr ApplyRewriteRules(EQ node);

  /*! \brief Rewrite rules for Equal comparisons
   *
   * These are separate from the VisitExpr_(const EQNode*) method, as
   * they may required from rewrites of LT, LE, or NE.
   */
  PrimExpr ApplyRewriteRules(Not node);

 private:
  CompareResult TryCompareUsingKnownInequalities(const PrimExpr& x, const PrimExpr& y);
  CompareResult TryCompareUsingConstIntBounds(const PrimExpr& x, const PrimExpr y);
  CompareResult TryComparisonOfProductAndSum(const PrimExpr& x, const PrimExpr& y);

  // Whether x >= val
  bool CanProveGreaterEqual(const PrimExpr& x, int64_t val) {
    return analyzer_->CanProveGreaterEqual(x, val);
  }
  // Whether x < val
  bool CanProveLess(const PrimExpr& x, int64_t val) { return analyzer_->CanProveLess(x, val); }
  // Whether x == val
  bool CanProveEqual(const PrimExpr& x, int64_t val) {
    // TODO(tqchen) refer back to super-analyzer.
    return TryCompare(x, val) == CompareResult::kEQ;
  }
  // Whether x is true
  bool CanProve(const PrimExpr& x) { return analyzer_->CanProve(x); }

  // Recursive rewrite x
  // we limit maximum depth of recursive rewrite allowed to
  // avoid infinite loop
  PrimExpr RecursiveRewrite(const PrimExpr& x) {
    stats_.num_recursive_rewrites++;
    if (recur_depth_ >= kMaxRecurDepth) return x;
    ++recur_depth_;
    stats_.max_recursive_depth = std::max(recur_depth_, stats_.max_recursive_depth);
    PrimExpr res = this->VisitExpr(x);
    --recur_depth_;
    return res;
  }

  template <typename TA>
  PConstWithTypeLike<TA> ZeroWithTypeLike(const Pattern<TA>& pattern) {
    return PConstWithTypeLike<TA>(pattern.derived(), 0);
  }

  template <typename TA>
  PConstWithTypeLike<TA> OneWithTypeLike(const Pattern<TA>& pattern) {
    return PConstWithTypeLike<TA>(pattern.derived(), 1);
  }
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_REWRITE_SIMPLIFY_H_
