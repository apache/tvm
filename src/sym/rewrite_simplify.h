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
#ifndef TVM_SYM_REWRITE_SIMPLIFY_H_
#define TVM_SYM_REWRITE_SIMPLIFY_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "const_fold.h"
#include "pattern_match.h"
#include "simplify_base.h"

namespace tvm {
namespace sym {

/* \brief Usage counters for RewriteSimplifier
 *
 * These are intended for debug and testing purposes, to ensure that
 * PrimExpr simplifications and TIR passes do not require an excessive
 */
struct RewriteSimplifierStatsNode : ffi::Object {
  int64_t nodes_visited{0};
  int64_t constraints_entered{0};
  int64_t rewrites_attempted{0};
  int64_t rewrites_performed{0};
  int64_t max_recursive_depth{0};
  int64_t num_recursive_rewrites{0};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteSimplifierStatsNode>()
        .def_ro("nodes_visited", &RewriteSimplifierStatsNode::nodes_visited)
        .def_ro("constraints_entered", &RewriteSimplifierStatsNode::constraints_entered)
        .def_ro("rewrites_attempted", &RewriteSimplifierStatsNode::rewrites_attempted)
        .def_ro("rewrites_performed", &RewriteSimplifierStatsNode::rewrites_performed)
        .def_ro("max_recursive_depth", &RewriteSimplifierStatsNode::max_recursive_depth)
        .def_ro("num_recursive_rewrites", &RewriteSimplifierStatsNode::num_recursive_rewrites);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sym.RewriteSimplifierStats", RewriteSimplifierStatsNode,
                                    ffi::Object);
};

struct RewriteSimplifierStats : ffi::ObjectRef {
  explicit RewriteSimplifierStats(RewriteSimplifierStatsNode data) {
    data_ = ffi::make_object<RewriteSimplifierStatsNode>(data);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(RewriteSimplifierStats, ffi::ObjectRef,
                                             RewriteSimplifierStatsNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RewriteSimplifierStatsNode);
};

/*!
 * \brief Rewrite-based simplifier.
 *
 * This class can be inheritated for other simplifiers.
 */
class RewriteSimplifier::Impl : public SimplifierBase {
 public:
  using SimplifierBase::Mutate;
  using SimplifierBase::Mutate_;

  explicit Impl(AnalyzerObj* parent) : SimplifierBase(parent) {}

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                               InplaceMode inplace_mode = InplaceMode::kDisallow) override;

  void Update(const Var& var, const PrimExpr& info, bool override_info);
  UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SubNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MulNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::DivNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::ModNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MinNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::NENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::AndNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::OrNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::NotNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) override;

  std::function<void()> EnterConstraint(const PrimExpr& constraint, bool is_assume);

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

  void CopyFrom(const Impl& other) {
    var_map_ = other.var_map_;
    literal_constraints_ = other.literal_constraints_;
    enabled_extensions_ = other.enabled_extensions_;
    maximum_rewrite_steps_ = other.maximum_rewrite_steps_;
  }

 protected:
  int64_t maximum_rewrite_steps_{0};
  RewriteSimplifierStatsNode stats_;

  void RecordAttemptedRewrite() { stats_.rewrites_attempted++; }
  void RecordRewrite() {
    stats_.rewrites_performed++;

    TVM_FFI_ICHECK(maximum_rewrite_steps_ <= 0 ||
                   stats_.rewrites_performed <= maximum_rewrite_steps_)
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
  bool CanInlineLet(const prim::LetNode* op);

  /*! \brief Internal function to apply constraints
   *
   * Tests whether the expression is known to be true or false based
   * on existing constraints.  If the expression or its negation
   * matches a constraint, return the boolean it should be replaced
   * with.  Otherwise, return false.
   */
  ffi::Optional<PrimExpr> TryMatchLiteralConstraint(const PrimExpr& expr) const;

  /*! \brief Rewrite rules for Less Than comparisons
   *
   * These are separate from the Mutate_(const LTNode*) method, as
   * they may required from rewrites of LT or LE.
   */
  PrimExpr ApplyRewriteRules(prim::LT node, InplaceMode inplace_mode);

  /*! \brief Rewrite rules for Equal comparisons
   *
   * These are separate from the Mutate_(const EQNode*) method, as
   * they may required from rewrites of LE or NE.
   */
  PrimExpr ApplyRewriteRules(prim::EQ node, InplaceMode inplace_mode);

  /*! \brief Rewrite rules for Equal comparisons
   *
   * These are separate from the Mutate_(const EQNode*) method, as
   * they may required from rewrites of LT, LE, or NE.
   */
  PrimExpr ApplyRewriteRules(prim::Not node, InplaceMode inplace_mode);

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
  bool CanProveGreaterEqual(const PrimExpr& x, const ffi::BigInt& val) {
    auto bound = val.as<int64_t>();
    return bound.has_value() && analyzer_->CanProveGreaterEqual(x, *bound);
  }
  bool CanProveLess(const PrimExpr& x, const ffi::BigInt& val) {
    auto bound = val.as<int64_t>();
    return bound.has_value() && analyzer_->CanProveLess(x, *bound);
  }
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
  PrimExpr RecursiveRewrite(const PrimExpr& x, InplaceMode inplace_mode) {
    stats_.num_recursive_rewrites++;
    if (recur_depth_ >= kMaxRecurDepth) return x;
    struct DepthGuard {
      int64_t& depth;
      explicit DepthGuard(int64_t& depth) : depth(depth) { ++depth; }
      ~DepthGuard() { --depth; }
    } depth_guard(recur_depth_);
    stats_.max_recursive_depth = std::max(recur_depth_, stats_.max_recursive_depth);
    return Mutate(x, inplace_mode).ValueOrUnchanged(x);
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

}  // namespace sym
}  // namespace tvm
#endif  // TVM_SYM_REWRITE_SIMPLIFY_H_
