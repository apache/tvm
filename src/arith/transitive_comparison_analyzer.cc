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
 * \file tvm/arith/transitive_comparison_analyzer.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>

#include <optional>
#include <vector>

#include "constraint_extract.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace tir;

class TransitiveComparisonAnalyzer::Impl {
 public:
  /* \brief Using previously specified knowns, compare the expressions provided
   *
   * \param lhs The left-hand side of the comparison
   *
   * \param rhs The right-hand side of the comparison
   *
   * \return The most specific result that can be proven about the
   * comparison.  If nothing can be proven, returns kUnknown.
   */
  CompareResult TryCompare(const PrimExpr& lhs, const PrimExpr& rhs) const;

  /*! \brief Bind a variable as being equal to a known expression
   *
   * \param var The variable of interest.
   * \param expr The bound expression
   * \param allow_override Whether to allow override of existing information.
   */
  void Bind(const tir::Var& var, const PrimExpr& expr, bool allow_override = false);

  /*! \brief Bind a variable as being within a specified range
   *
   * \param var The variable of interest.
   * \param range The known range
   * \param allow_override Whether to allow override of existing information.
   */
  void Bind(const tir::Var& var, const Range& expr, bool allow_override = false);

  /*!
   * \brief Update the internal state to enter constraint.
   * \param constraint A constraint expression.
   *
   * \return An exit function that must be called to cleanup.  May be
   * `nullptr`, if no cleanup is required.
   */
  std::function<void()> EnterConstraint(const PrimExpr& expr);

 private:
  /* \brief Internal representation of a PrimExpr
   *
   * The Key enum serves two purposes.
   *
   * 1. Providing efficiency, as compared to a PrimExpr.  Two keys are
   *    equal if and only if the corresponding PrimExprs would satisfy
   *    ExprDeepEqual.  This allows two expressions to be checked for
   *    equivalency, without requiring a call to ExprDeepEqual for
   *    each comparison.
   *
   * 2. Providing type-safety, as compared to using `size_t` directly.
   *    Requiring an explicit conversion from an integer to a Key
   *    prevents accidental comparisons, especially if both loop
   *    iterators and Keys are used in the same scope.
   *
   * A Key should only be obtained using the methods `ExprToKey` and
   * `ExprToPreviousKey`.
   */
  enum class Key : size_t {};

  /*! \brief Convert an expression to internal representation
   *
   * If the expression has previously been converted to the internal
   * representation, returns the same Key as has been used previously.
   * Otherwise, generate and return a new Key.
   *
   * \param expr The PrimExpr to be converted
   *
   * \returns The Key representing the expression
   *
   * \see ExprToPreviousKey
   */
  Key ExprToKey(const PrimExpr& expr);

  /*! \brief Convert an expression to internal representation
   *
   * If the expression has previously been converted to the internal
   * representation, returns the same Key as has been used previously.
   * Otherwise, return `std::nullopt`.
   *
   * \param expr The PrimExpr to be converted
   *
   * \returns The Key representing the expression, if one exists.
   *
   * \see ExprToKey
   */
  std::optional<Key> ExprToPreviousKey(const PrimExpr& expr) const;

  /*! \brief The mapping from expression to Key
   *
   * Should not be used directly.  Instead, use the helper functions
   * `ExprToKey` and `ExprToPreviousKey`.
   *
   * \see ExprToKey
   * \see ExprToPreviousKey
   */
  std::unordered_map<PrimExpr, Key, StructuralHash, StructuralEqual> expr_to_key;

  /*! \brief Internal representation of a comparison operator */
  struct Comparison {
    /*! \brief Construct a comparison that represents `lhs OP rhs +
     * offset`, where the operation is specified by the CompareResult.
     */
    Comparison(Key lhs, Key rhs, int64_t offset, CompareResult result);

    /*! \brief Utility function to validate that all GT and LT results
     *  have been normalized out
     */
    bool IsNormalized() const;

    /*! \brief Move the specified expression to the LHS.
     *
     * \param new_lhs The argument that should be moved to the LHS of the
     * comparison.
     *
     * \return If possible, returns a comparison that is equivalent to
     * the current comparison, but with the specified LHS.  If not
     * possible, returns nullopt.
     */
    std::optional<Comparison> WithLHS(Key new_lhs) const;

    /*! \brief Create the negation of the current comparison */
    Comparison Negated() const;

    /*! \brief Check the this comparison implies
     *
     * Returns true if this comparison being true implies that the
     * other comparison must also be true.  Returns false if the other
     * comparison cannot be shown to be true.
     */
    bool Implies(const Comparison& other) const;

    // The LHS of the comparison
    Key lhs_;

    // The RHS of the comparison, not including any constant offset.
    Key rhs_;

    // Additive offset on rhs
    int64_t offset_{0};

    // The comparison operator.
    CompareResult result_{CompareResult::kInconsistent};
  };

  /*! \brief Generate a Comparison representing the given expression */
  std::optional<Comparison> FromExpr(const PrimExpr& expr);

  /*! \brief Utility function used by Bind and EnterConstraint
   *
   * \param expr The comparison expression, to be converted into
   * internal Comparison objects.
   *
   * \param vec The vector to which the Comparison objects should be
   * appended.
   */
  void AddKnown(const PrimExpr& expr, std::vector<Comparison>* vec);

  /*! \brief Attempt to compare the expressions, starting at the lhs.
   *
   * Perform a depth-first search through the space of known
   * expressions, starting at the LHS of a comparison.  In this
   * search, each expression is a node of a graph, and each known
   * comparison is an edge of the graph.
   *
   * For example, suppose we have previous knowns of (A<=B), (B<=C+1)
   * and (C<=D-5).  The expressions [A,B,C,D] are the nodes of the
   * search space.  Each comparison is an edge connecting two
   * expressions, such as (B<=C+1) connecting the expressions B and D.
   * If we are attempting to compare expressions A and D, a search
   * starting at expression A could follow each edge until reaching
   * expression D, then combine the comparisons that compose the path
   * into the expression A<=D-4.
   *
   * \param lhs The left-hand side of the comparison
   *
   * \param rhs The right-hand side of the comparison
   *
   * \return The result of the comparison
   */
  CompareResult DFSFromLHS(Key lhs_key, Key rhs_key, int64_t offset, const PrimExpr& lhs,
                           const PrimExpr& rhs) const;

  /*! \brief Previous Range bindings
   *
   * Tracked separatedly to handle the `allow_override` option used by
   * all sub-analyzers when binding variables.
   */
  Map<Var, Range> prev_bindings_;

  /*! \brief Known comparisons based on definitionally-true statements
   *
   * For example, a Let binding, or the range of an iterator.  These
   * known statements are always true, based on the definition site of
   * the variable.  e.g. A loop iterator may never exceed the bounds
   * of its loop.
   */
  std::vector<Comparison> knowns_;

  /*! \brief Known comparisons based on scoped conditions
   *
   * For example, the condition of an IfThenElse.  These known
   * statements may only be used within the scope of the statement
   * from which they were derived.  e.g. After exiting an IfThenElse,
   * the condition may no longer be true.
   */
  std::vector<Comparison> scoped_knowns_;
};

namespace {

// Internal utility, return the CompareResult resulting from swapping
// the left-hand side with the right-hand side.
CompareResult Reverse(CompareResult res) {
  switch (res) {
    case CompareResult::kInconsistent:
      return CompareResult::kInconsistent;
    case CompareResult::kEQ:
      return CompareResult::kEQ;
    case CompareResult::kLT:
      return CompareResult::kGT;
    case CompareResult::kLE:
      return CompareResult::kGE;
    case CompareResult::kGT:
      return CompareResult::kLT;
    case CompareResult::kGE:
      return CompareResult::kLE;
    case CompareResult::kNE:
      return CompareResult::kNE;
    case CompareResult::kUnknown:
      return CompareResult::kUnknown;
    default:
      LOG(FATAL) << "Invalid CompareResult: " << static_cast<int>(res);
      return CompareResult::kInconsistent;
  }
}

// Internal utility, return the CompareResult resulting from negating
// the comparison.
CompareResult Negate(CompareResult res) {
  switch (res) {
    case CompareResult::kInconsistent:
      return CompareResult::kInconsistent;
    case CompareResult::kUnknown:
      return CompareResult::kUnknown;
    default:
      return CompareResult(~static_cast<int>(res) & static_cast<int>(CompareResult::kUnknown));
  }
}

// Internal utility, extract constant offsets out of the two sides of
// a comparison.  Given lhs and rhs, return a tuple of three elements
// (lhs_inner, rhs_inner, offset), such that (lhs OP rhs) and
// (lhs_inner OP rhs_inner + offset) are equivalent.
std::tuple<PrimExpr, PrimExpr, int64_t> ExtractOffsets(const PrimExpr& lhs, const PrimExpr& rhs) {
  auto extract_offset = [](const PrimExpr& expr) -> std::pair<PrimExpr, int64_t> {
    PVar<PrimExpr> x;
    PVar<IntImm> c;
    if ((x + c).Match(expr)) {
      return {x.Eval(), c.Eval()->value};
    } else if ((x - c).Match(expr)) {
      return {x.Eval(), -c.Eval()->value};
    } else if (c.Match(expr)) {
      return {0, c.Eval()->value};
    } else {
      return {expr, 0};
    }
  };

  auto lhs_split = extract_offset(lhs);
  auto rhs_split = extract_offset(rhs);
  return {lhs_split.first, rhs_split.first, rhs_split.second - lhs_split.second};
}

}  // namespace

std::optional<TransitiveComparisonAnalyzer::Impl::Comparison>
TransitiveComparisonAnalyzer::Impl::FromExpr(const PrimExpr& expr) {
  CompareResult res;
  PVar<PrimExpr> x, y;
  if ((x <= y).Match(expr)) {
    res = CompareResult::kLE;
  } else if ((x >= y).Match(expr)) {
    res = CompareResult::kGE;
  } else if ((x < y).Match(expr)) {
    res = CompareResult::kLT;
  } else if ((x > y).Match(expr)) {
    res = CompareResult::kGT;
  } else if ((x == y).Match(expr)) {
    res = CompareResult::kEQ;
  } else if ((x != y).Match(expr)) {
    res = CompareResult::kNE;
  } else {
    return std::nullopt;
  }

  PrimExpr lhs_expr = x.Eval();
  PrimExpr rhs_expr = y.Eval();

  if (lhs_expr.as<IntImmNode>() && rhs_expr.as<IntImmNode>()) {
    return std::nullopt;
  }

  auto [lhs, rhs, offset] = ExtractOffsets(lhs_expr, rhs_expr);
  Key lhs_key = ExprToKey(lhs);
  Key rhs_key = ExprToKey(rhs);

  return Comparison(lhs_key, rhs_key, offset, res);
}

TransitiveComparisonAnalyzer::Impl::Comparison::Comparison(Key lhs, Key rhs, int64_t offset,
                                                           CompareResult result)
    : lhs_(lhs), rhs_(rhs), offset_(offset), result_(result) {
  // Normalize the comparison to remove LT and GT expressions,
  // reducing the number of operators that must be handled later.  By
  // eliminating LT and GT, instead of eliminating LE or GE, a
  // potential off-by-one error is avoided.
  //
  // For floating-point numbers, (x < y + c1) and (y < z + c2) implies
  // that (x < z + (c1 + c2)).  For integer types, which the
  // TransitiveComparisonAnalyzer is intended for use with integers,
  // LT or GT can give a tighter constraint, though with a less
  // convenient symmetry.
  //
  // i < j + c1, j < k + c2
  // i <= j + c1 - 1, j <= k + c2 - 1
  // i + 1 - c1 <= j, j <= k + c2 - 1
  // i + 1 - c1 <= k + c2 - 1
  // i <= k + c1 + c2 - 2
  // i < k + (c1 + c2 - 1)
  //
  // By always working with LE and GE comparisons, we avoid needing to
  // handle the offset of one that would be introduced by LT and GT at
  // all points of use.  The only point of use for LT and GT is when
  // normalizing comparisons (i.e. this constructor).

  if (result_ == CompareResult::kLT) {
    result_ = CompareResult::kLE;
    offset_ -= 1;
  }
  if (result_ == CompareResult::kGT) {
    result_ = CompareResult::kGE;
    offset_ += 1;
  }
}

std::optional<TransitiveComparisonAnalyzer::Impl::Key>
TransitiveComparisonAnalyzer::Impl::ExprToPreviousKey(const PrimExpr& expr) const {
  auto it = expr_to_key.find(expr);
  if (it != expr_to_key.end()) {
    return it->second;
  } else {
    return std::nullopt;
  }
}

TransitiveComparisonAnalyzer::Impl::Key TransitiveComparisonAnalyzer::Impl::ExprToKey(
    const PrimExpr& expr) {
  if (auto prev = ExprToPreviousKey(expr)) {
    return prev.value();
  } else {
    Key new_key = Key(expr_to_key.size());
    expr_to_key[expr] = new_key;
    return new_key;
  }
}

bool TransitiveComparisonAnalyzer::Impl::Comparison::IsNormalized() const {
  // These < and > should be removed during normalization.  See the
  // `Comparison::Comparison` constructor for further details.
  return result_ != CompareResult::kLT && result_ != CompareResult::kGT;
}

std::optional<TransitiveComparisonAnalyzer::Impl::Comparison>
TransitiveComparisonAnalyzer::Impl::Comparison::WithLHS(Key new_lhs) const {
  if (new_lhs == lhs_) {
    return *this;
  } else if (new_lhs == rhs_) {
    return Comparison(rhs_, lhs_, -offset_, Reverse(result_));
  } else {
    return std::nullopt;
  }
}

TransitiveComparisonAnalyzer::Impl::Comparison
TransitiveComparisonAnalyzer::Impl::Comparison::Negated() const {
  return Comparison(lhs_, rhs_, offset_, Negate(result_));
}

bool TransitiveComparisonAnalyzer::Impl::Comparison::Implies(
    const TransitiveComparisonAnalyzer::Impl::Comparison& other) const {
  ICHECK(lhs_ == other.lhs_);
  ICHECK(rhs_ == other.rhs_);
  ICHECK(IsNormalized());
  ICHECK(other.IsNormalized());

  if (result_ == other.result_ && offset_ == other.offset_) {
    // if c1 == c2, x != y + c1 => x != y + c2
    // if c1 == c2, x == y + c1 => x == y + c2
    return true;
  }

  if (other.result_ == CompareResult::kLE && offset_ <= other.offset_) {
    if (result_ == CompareResult::kEQ || result_ == CompareResult::kLE) {
      // if c1 <= c2, x <= y + c1 => x <= y + c2
      // if c1 <= c2, x == y + c1 => x <= y + c2
      return true;
    }
  }

  if (other.result_ == CompareResult::kGE && offset_ >= other.offset_) {
    if (result_ == CompareResult::kEQ || result_ == CompareResult::kGE) {
      // if c1 >= c2, x == y + c1 => x >= y + c2
      // if c1 >= c2, x >= y + c1 => x >= y + c2
      return true;
    }
  }

  if (other.result_ == CompareResult::kNE) {
    if (result_ == CompareResult::kEQ && offset_ != other.offset_) {
      // if c1 != c2, x == y + c1 => x != y + c2
      return true;
    }

    if (result_ == CompareResult::kLE && offset_ < other.offset_) {
      // if c1 < c2, x <= y + c1 => x < y + c2 => x != y + c2
      return true;
    }

    if (result_ == CompareResult::kGE && offset_ > other.offset_) {
      // if c1 != c2, x >= y + c1 => x > y + c2 => x != y + c2
      return true;
    }
  }

  return false;
}

TransitiveComparisonAnalyzer::TransitiveComparisonAnalyzer() : impl_(std::make_unique<Impl>()) {}
TransitiveComparisonAnalyzer::~TransitiveComparisonAnalyzer() {}

CompareResult TransitiveComparisonAnalyzer::TryCompare(const PrimExpr& lhs, const PrimExpr& rhs) {
  return impl_->TryCompare(lhs, rhs);
}

void TransitiveComparisonAnalyzer::Bind(const Var& var, const PrimExpr& expr, bool allow_override) {
  impl_->Bind(var, expr, allow_override);
}
void TransitiveComparisonAnalyzer::Bind(const Var& var, const Range& range, bool allow_override) {
  impl_->Bind(var, range, allow_override);
}

std::function<void()> TransitiveComparisonAnalyzer::EnterConstraint(const PrimExpr& constraint) {
  return impl_->EnterConstraint(constraint);
}

void TransitiveComparisonAnalyzer::Impl::AddKnown(const PrimExpr& expr,
                                                  std::vector<Comparison>* vec) {
  for (const auto& subexpr : ExtractConstraints(expr)) {
    if (tir::SideEffect(expr) <= tir::CallEffectKind::kPure) {
      if (auto cmp = FromExpr(subexpr)) {
        vec->push_back(cmp.value());
      }
    }
  }
}

void TransitiveComparisonAnalyzer::Impl::Bind(const tir::Var& var, const Range& range,
                                              bool allow_override) {
  auto it = prev_bindings_.find(var);
  if (it != prev_bindings_.end()) {
    ExprDeepEqual expr_equal;
    bool differs_from_previous = !expr_equal(range->min, (*it).second->min) ||
                                 !expr_equal(range->extent, (*it).second->extent);
    if (differs_from_previous) {
      ICHECK(allow_override) << "Binding of variable " << var << " as " << range
                             << " conflicts with previous binding as " << (*it).second;
      if (auto key = ExprToPreviousKey(var)) {
        knowns_.erase(std::remove_if(knowns_.begin(), knowns_.end(),
                                     [&](const auto& known) { return known.lhs_ == key.value(); }),
                      knowns_.end());
      }
    }
  }

  prev_bindings_.Set(var, range);

  if (is_const_int(range->extent, 1)) {
    AddKnown(var == range->min, &knowns_);
  } else {
    AddKnown(var >= range->min, &knowns_);
    AddKnown(var < range->min + range->extent, &knowns_);
  }
}

void TransitiveComparisonAnalyzer::Impl::Bind(const tir::Var& var, const PrimExpr& expr,
                                              bool allow_override) {
  Bind(var, Range::FromMinExtent(expr, 1), allow_override);
}

std::function<void()> TransitiveComparisonAnalyzer::Impl::EnterConstraint(const PrimExpr& expr) {
  size_t old_literal_size = scoped_knowns_.size();
  AddKnown(expr, &scoped_knowns_);
  size_t new_literal_size = scoped_knowns_.size();

  auto frecover = [old_literal_size, new_literal_size, this]() {
    ICHECK_EQ(scoped_knowns_.size(), new_literal_size);
    scoped_knowns_.erase(scoped_knowns_.begin() + old_literal_size, scoped_knowns_.end());
  };
  return frecover;
}

CompareResult TransitiveComparisonAnalyzer::Impl::TryCompare(const PrimExpr& lhs_expr,
                                                             const PrimExpr& rhs_expr) const {
  // Currently only supports integer checks
  if (!lhs_expr.dtype().is_int() || !rhs_expr.dtype().is_int()) {
    return CompareResult::kUnknown;
  }

  // Bail out early if possible.  This int check should have been
  // constant-folded earlier, so this check shouldn't occur.
  auto* x_int = lhs_expr.as<IntImmNode>();
  auto* y_int = rhs_expr.as<IntImmNode>();
  if (x_int && y_int) {
    if (x_int->value < y_int->value) {
      return CompareResult::kLT;
    } else if (x_int->value > y_int->value) {
      return CompareResult::kGT;
    } else {
      return CompareResult::kEQ;
    }
  }

  auto [lhs, rhs, offset] = ExtractOffsets(lhs_expr, rhs_expr);
  auto lhs_key = ExprToPreviousKey(lhs);
  auto rhs_key = ExprToPreviousKey(rhs);

  if (!lhs_key.has_value() || !rhs_key.has_value()) {
    return CompareResult::kUnknown;
  }

  auto from_lhs = DFSFromLHS(lhs_key.value(), rhs_key.value(), offset, lhs, rhs);
  auto from_rhs = Reverse(DFSFromLHS(rhs_key.value(), lhs_key.value(), -offset, rhs, lhs));
  auto output = from_lhs & from_rhs;

  return output;
}

CompareResult TransitiveComparisonAnalyzer::Impl::DFSFromLHS(Key lhs_key_input, Key rhs_key_input,
                                                             int64_t offset_input,
                                                             const PrimExpr& lhs_input,
                                                             const PrimExpr& rhs_input) const {
  Key lhs_key = lhs_key_input;
  Key rhs_key = rhs_key_input;
  int64_t offset = offset_input;

  // Everything in `to_visit` has lhs as its lhs.
  std::unordered_set<Key> seen;
  std::unordered_set<Key> to_visit;
  std::unordered_map<Key, std::vector<Comparison>> compared_to_x;

  // Utility function to add a new known statement
  auto declare_known = [&](Comparison cmp) {
    std::vector<Comparison>& knowns = compared_to_x[cmp.rhs_];

    // The comparison adds no new information, no modification
    // required.
    for (auto& prev_known : knowns) {
      if (prev_known.Implies(cmp)) {
        return;
      }
    }

    // New information may require visiting a new expression.
    if (cmp.rhs_ != rhs_key && !seen.count(cmp.rhs_)) {
      to_visit.insert(cmp.rhs_);
      seen.insert(cmp.rhs_);
    }

    // This comparison is a stronger version of a previous constraint.
    // Therefore, replace the old version entirely.
    for (auto& prev_known : knowns) {
      if (cmp.Implies(prev_known)) {
        prev_known = cmp;
        return;
      }
    }

    // Neither a superset nor a subset of previously known
    // constraints, must be tracked separately.
    knowns.push_back(cmp);
  };

  // Initialize the search based on any known (in)equalities that use
  // the LHS of the comparison.
  for (const auto& known : knowns_) {
    if (auto normalized = known.WithLHS(lhs_key)) {
      declare_known(normalized.value());
    }
  }
  for (const auto& known : scoped_knowns_) {
    if (auto normalized = known.WithLHS(lhs_key)) {
      declare_known(normalized.value());
    }
  }

  // Walk through the space of all comparisons that can be made with
  // LHS.
  while (to_visit.size()) {
    Key middle_key = *to_visit.begin();
    to_visit.erase(to_visit.begin());

    std::vector<Comparison>& prev_knowns_using_middle = compared_to_x.at(middle_key);
    ICHECK(compared_to_x.count(middle_key));

    std::vector<Comparison> new_knowns_using_lhs;

    auto attempt_transitive = [&](Comparison cmp) {
      ICHECK(cmp.IsNormalized());

      Key right_key = cmp.rhs_;

      if (right_key == lhs_key) {
        return;
      }

      for (const auto& prev : prev_knowns_using_middle) {
        CompareResult new_result = CompareResult::kUnknown;
        int64_t new_offset = prev.offset_ + cmp.offset_;

        if (prev.result_ == CompareResult::kEQ) {
          // x == y + c1 && y OP z + c2, x OP z + (c1 + c2)
          new_result = cmp.result_;
        } else if (cmp.result_ == CompareResult::kEQ) {
          // x OP y + c1 && y == z + c2, x OP z + (c1 + c2)
          new_result = prev.result_;
        } else if (prev.result_ == cmp.result_ &&
                   (prev.result_ == CompareResult::kLE || prev.result_ == CompareResult::kGE)) {
          // x <= y + c1 && y <= z + c2, x <= z + (c1 + c2)
          // x >= y + c1 && y >= z + c2, x >= z + (c1 + c2)
          //
          // This condition is much simpler to write than the
          // equivalent handling of < or of >, which is why the
          // inequalities are normalized to <= and to >=.  See
          // `TransitiveComparisonAnalyzer::Impl::Comparison::Comparison`
          // for further details.
          new_result = prev.result_;
        }

        if (new_result != CompareResult::kUnknown) {
          Comparison new_known(lhs_key, right_key, new_offset, new_result);
          new_knowns_using_lhs.push_back(new_known);
        }
      }
    };

    // Attempt to prove a new comparison using one of the original
    // known comparisons.  We want to find a known such that
    // `(LHS OP1 middle) && (middle OP2 right)` can be simplified
    // into `(LHS OP3 right)`.
    //
    // Note: The right side is this step is not necessarily the RHS of
    // the comparison we're trying to prove, as we may need to find
    // intermediate comparisons first.  For example, if we know that
    // `a<=b`, `b<=c`, and `c<=d`, and we wish to prove that `a<=d`,
    // we must first combine `a<=b` and `b<=c` into `a<=c`.  During
    // this first step, `b` is the "middle" and `c` is the "right".
    // The next step can then combind `a<=c` and `c<=d` into `a<=d`.
    for (const auto& known : knowns_) {
      if (auto cmp = known.WithLHS(middle_key)) {
        attempt_transitive(cmp.value());
      }
    }

    for (const auto& known : scoped_knowns_) {
      if (auto cmp = known.WithLHS(middle_key)) {
        attempt_transitive(cmp.value());
      }
    }

    // Collect together all new knowns, marking new nodes for visiting
    // as needed.
    for (const auto& new_known : new_knowns_using_lhs) {
      declare_known(new_known);
    }
  }

  // It's possible that we don't have any transitive comparisons that
  // can prove something about LHS and RHS.
  auto it = compared_to_x.find(rhs_key);
  if (it == compared_to_x.end()) {
    return CompareResult::kUnknown;
  }

  const std::vector<Comparison>& known_between_lhs_and_rhs = it->second;

  // Just because we found a comparison involving LHS and RHS doesn't
  // mean that it's useful.  e.g. Knowing that `x < y` doesn't let us
  // prove whether `x + 5 < y`.
  CompareResult result = CompareResult::kUnknown;
  for (const auto& known : known_between_lhs_and_rhs) {
    switch (known.result_) {
      case CompareResult::kInconsistent:
        result = CompareResult::kInconsistent;
        break;

      case CompareResult::kEQ:
        if (offset == known.offset_) {
          result = result & CompareResult::kEQ;
        } else {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kLE:
        if (known.offset_ < offset) {
          result = result & CompareResult::kLT;
        } else if (known.offset_ <= offset) {
          result = result & CompareResult::kLE;
        }
        break;

      case CompareResult::kGE:
        if (known.offset_ > offset) {
          result = result & CompareResult::kGT;
        } else if (known.offset_ >= offset) {
          result = result & CompareResult::kGE;
        }
        break;

      case CompareResult::kNE:
        if (offset == known.offset_) {
          result = result & CompareResult::kNE;
        }
        break;

      case CompareResult::kUnknown:
        break;

      case CompareResult::kGT:
      case CompareResult::kLT:
        LOG(FATAL) << "Internal error, normalized comparisons should only include <= and >=";
        return CompareResult::kInconsistent;

      default:
        LOG(FATAL) << "Invalid CompareResult: " << static_cast<int>(known.result_);
        return CompareResult::kInconsistent;
    }
  }

  return result;
}

}  // namespace arith
}  // namespace tvm
