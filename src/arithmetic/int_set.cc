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
 * \file int_set.cc
 * \brief The integer set functions
 */
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/api_registry.h>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include "int_set.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

Expr SymbolicLimits::pos_inf_ = Var("pos_inf", Handle());
Expr SymbolicLimits::neg_inf_ = Var("neg_inf", Handle());

IntervalSet::IntervalSet(Expr min_value, Expr max_value) {
  auto node = make_node<IntervalSetNode>();
  node->min_value = std::move(min_value);
  node->max_value = std::move(max_value);
  node_ = std::move(node);
}

IntervalSet MakeIntervalSet(Expr min_value, Expr max_value) {
  return IntervalSet(min_value, max_value);
}

TVM_REGISTER_API("arith._make_IntervalSet")
.set_body_typed(MakeIntervalSet);


IntervalSet Intersect(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  Expr max_value = min(a->max_value, b->max_value);
  Expr min_value = max(a->min_value, b->min_value);
  if ((max_value.type().is_int() || max_value.type().is_uint()) &&
      (min_value.type().is_int() || min_value.type().is_uint()) &&
      analyzer->CanProveGreaterEqual(min_value - max_value, 1)) {
    return IntervalSet::Empty();
  } else {
    return IntervalSet(min_value, max_value);
  }
}

IntervalSet Union(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  Expr max_value = max(a->max_value, b->max_value);
  Expr min_value = min(a->min_value, b->min_value);
  return IntervalSet(min_value, max_value);
}

// type traits
template<typename OP>
struct is_logical_op {
  static const bool value = false;
};

#define TVM_DECLARE_LOGICAL_OP(OP)              \
  template<>                                    \
  struct is_logical_op<ir::OP> {                \
    static const bool value = true;             \
  };

TVM_DECLARE_LOGICAL_OP(And);
TVM_DECLARE_LOGICAL_OP(Or);
TVM_DECLARE_LOGICAL_OP(EQ);
TVM_DECLARE_LOGICAL_OP(NE);
TVM_DECLARE_LOGICAL_OP(GE);
TVM_DECLARE_LOGICAL_OP(GT);
TVM_DECLARE_LOGICAL_OP(LE);
TVM_DECLARE_LOGICAL_OP(LT);
TVM_DECLARE_LOGICAL_OP(Not);

/*!
 * \brief Combine two interval set under arithmetic operations.
 * \note this can possibly relax the set.
 */
template<typename Op>
inline IntervalSet Combine(Analyzer* analyzer,
                           IntervalSet a,
                           IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    Expr res = TryConstFold<Op>(a->min_value, b->min_value);
    if (!res.defined()) res = Op::make(a->min_value, b->min_value);
    return IntervalSet::SinglePoint(res);
  }
  if (is_logical_op<Op>::value) {
    return IntervalSet(make_const(a->min_value.type(), 0),
                       make_const(a->min_value.type(), 1));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (a->IsEverything()) return a;
  if (b->IsEverything()) return b;
  return IntervalSet::Everything();
}

template<>
inline IntervalSet Combine<ir::Add>(Analyzer* analyer,
                                    IntervalSet a,
                                    IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value + b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  Expr min_value =
      a->HasLowerBound() && b->HasLowerBound() ?
      a->min_value + b->min_value : neg_inf();
  Expr max_value =
      a->HasUpperBound() && b->HasUpperBound() ?
      a->max_value + b->max_value : pos_inf();
  return IntervalSet(min_value, max_value);
}

template<>
inline IntervalSet Combine<ir::Sub>(Analyzer* analyer,
                                    IntervalSet a,
                                    IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value - b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  Expr min_value =
      a->HasLowerBound() && b->HasUpperBound() ?
      a->min_value - b->max_value : neg_inf();
  Expr max_value =
      a->HasUpperBound() && b->HasLowerBound() ?
      a->max_value - b->min_value : pos_inf();
  return IntervalSet(min_value, max_value);
}


template<>
inline IntervalSet Combine<ir::Mul>(Analyzer* analyzer,
                                    IntervalSet a,
                                    IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value * b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (a->IsSinglePoint()) {
    std::swap(a, b);
  }
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) return b;
    if (is_one(b->min_value)) return a;
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      Expr min_value = a->HasLowerBound() ? a->min_value * b->min_value : neg_inf();
      Expr max_value = a->HasUpperBound() ? a->max_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      Expr min_value = a->HasUpperBound() ? a->max_value * b->min_value : neg_inf();
      Expr max_value = a->HasLowerBound() ? a->min_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using ir::Select;
      Expr sign = b->min_value >= make_zero(b->min_value.type().element_of());
      Expr e1 = a->min_value * b->min_value;
      Expr e2 = a->max_value * b->min_value;
      return IntervalSet(Select::make(sign, e1, e2), Select::make(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mul";
  return IntervalSet::Everything();
}

template<>
inline IntervalSet Combine<ir::Div>(Analyzer* analyzer,
                                    IntervalSet a,
                                    IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value / b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) {
      LOG(FATAL) << "Divide by zero in CombineInterval Div";
    }
    if (is_one(b->min_value)) return a;
    // no relaxation is needed in here due to set is inclusive
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      Expr min_value = a->HasLowerBound() ? a->min_value / b->min_value : neg_inf();
      Expr max_value = a->HasUpperBound() ? a->max_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      Expr min_value = a->HasUpperBound() ? a->max_value / b->min_value : neg_inf();
      Expr max_value = a->HasLowerBound() ? a->min_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using ir::Select;
      Expr sign = b->min_value >= make_zero(b->min_value.type().element_of());
      Expr e1 = a->min_value / b->min_value;
      Expr e2 = a->max_value / b->min_value;
      return IntervalSet(Select::make(sign, e1, e2), Select::make(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template<>
inline IntervalSet Combine<ir::Mod>(Analyzer* analyzer,
                                    IntervalSet a,
                                    IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value % b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const Expr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    // We need to add more bound constraints throughout the code.
    // The logic below assumes a is non-negative, which usually
    // is the case of our application.
    // TODO(tqchen): add bound constraints for a.
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      return IntervalSet(make_zero(divisor.type()), divisor - 1);
    } else {
      Expr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}


template<>
inline IntervalSet Combine<ir::FloorDiv>(Analyzer* analyzer,
                                         IntervalSet a,
                                         IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(floordiv(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) {
      LOG(FATAL) << "Divide by zero in CombineInterval Div";
    }
    if (is_one(b->min_value)) return a;
    // no relaxation is needed in here due to set is inclusive
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      Expr min_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : neg_inf();
      Expr max_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      Expr min_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : neg_inf();
      Expr max_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using ir::Select;
      Expr sign = b->min_value >= make_zero(b->min_value.type().element_of());
      Expr e1 = floordiv(a->min_value, b->min_value);
      Expr e2 = floordiv(a->max_value, b->min_value);
      return IntervalSet(Select::make(sign, e1, e2), Select::make(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template<>
inline IntervalSet Combine<ir::FloorMod>(Analyzer* analyzer,
                                         IntervalSet a,
                                         IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(floormod(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const Expr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      return IntervalSet(make_zero(divisor.type()), divisor - 1);
    } else {
      Expr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}

template<>
inline IntervalSet Combine<ir::Max>(Analyzer* analzyer,
                                    IntervalSet a,
                                    IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(max(a->min_value,  b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(max(a->min_value, b->min_value),
                     max(a->max_value, b->max_value));
}

template<>
inline IntervalSet Combine<ir::Min>(Analyzer* analzyer,
                                    IntervalSet a,
                                    IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(min(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(min(a->min_value, b->min_value),
                     min(a->max_value, b->max_value));
}

// internal helper function to get an interval set
IntervalSet ToIntervalSet(IntSet set) {
  if (auto* node = set.as<IntervalSetNode>()) {
    return GetRef<IntervalSet>(node);
  }
  DLOG(INFO) << "cannot resolve int set " << set;
  return IntervalSet::Everything();
}

using namespace ir;

// Simplified version of int set evaluator that operates on IntervalSet
// We might use better set analysis in the future to replace the intervalset.
class IntervalSetEvaluator :
      public ExprFunctor<IntervalSet(const Expr&)> {
 public:
  IntervalSetEvaluator(Analyzer* analyzer,
                       const Map<Var, IntSet>& dom_map,
                       bool eval_vec = false)
      : analyzer_(analyzer),
        dom_map_(dom_map),
        eval_vec_(eval_vec) {
  }

  IntervalSet Eval(const Expr& val) {
    return this->VisitExpr(val);
  }
  // evaluate and relax the set
  IntervalSet Eval(IntervalSet val) {
    // avoid recursive indefinite recursive expansion.
    if (static_cast<size_t>(recur_depth_) >= dom_map_.size()) return val;
    ++recur_depth_;
    IntervalSet min_set = this->Eval(val->min_value);
    IntervalSet max_set = this->Eval(val->max_value);
    --recur_depth_;
    return IntervalSet(min_set->min_value, max_set->max_value);
  }

  IntervalSet VisitExpr_(const IntImm* op) final {
    return IntervalSet::SinglePoint(GetRef<Expr>(op));
  }

  IntervalSet VisitExpr_(const UIntImm* op) final {
    return IntervalSet::SinglePoint(GetRef<Expr>(op));
  }

  IntervalSet VisitExpr_(const Variable* op) final {
    Var var = GetRef<Var>(op);
    auto it = dom_map_.find(var);
    if (it != dom_map_.end()) {
      IntervalSet res = ToIntervalSet((*it).second);
      if (res->min_value.same_as(var) &&
          res->max_value.same_as(var)) {
        return res;
      }
      // recursively evaluate mapped result
      // in case the domain contains variables to be relaxed.
      return Eval(res);
    } else {
      return IntervalSet::SinglePoint(var);
    }
  }

  IntervalSet VisitExpr_(const Add* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Sub* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Mul* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Div* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Mod* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const FloorDiv* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const FloorMod* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Min* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Max* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const EQ* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const NE* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const LT* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const LE* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const GT* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const GE* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const And* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Or* op) final {
    return VisitBinaryExpr_(op);
  }

  IntervalSet VisitExpr_(const Ramp* op) final {
    CHECK(eval_vec_);
    IntervalSet base = Eval(op->base);
    PVar<Integer> stride;
    if (stride.Match(op->stride)) {
      Type t = op->base.type();
      int64_t vstride = stride.Eval()->value;
      if (vstride> 0) {
        return Combine<Add>(
            analyzer_,
            base,
            IntervalSet(make_zero(t), make_const(t, vstride * op->lanes - 1)));
      } else {
        return Combine<Add>(
            analyzer_,
            base,
            IntervalSet(make_const(t, vstride * op->lanes + 1), make_zero(t)));
      }
    }
    DLOG(WARNING) << "cannot evaluate set on expression " << GetRef<Expr>(op);
    return IntervalSet::Everything();
  }

  IntervalSet VisitExpr_(const Broadcast* op) final {
    CHECK(eval_vec_);
    return VisitExpr(op->value);
  }

  IntervalSet VisitExpr_(const Select* op) final {
    IntervalSet true_set = this->Eval(op->true_value);
    IntervalSet false_set = this->Eval(op->false_value);
    return Union(analyzer_, false_set, true_set);
  }

  IntervalSet VisitExprDefault_(const Node* op) final {
    DLOG(WARNING) << "cannot evaluate set type " << op->type_key();
    return IntervalSet::Everything();
  }

 private:
  // whether set is exactly single point that equals value.
  bool MatchPoint(const IntervalSet& set,
                  const Expr& value) const {
    return set->min_value.same_as(value) && set->max_value.same_as(value);
  }

  template<typename T>
  inline IntervalSet VisitBinaryExpr_(const T* op) {
    IntervalSet a = this->Eval(op->a);
    IntervalSet b = this->Eval(op->b);
    if (MatchPoint(a, op->a) && MatchPoint(b, op->b)) {
      return IntervalSet::SinglePoint(GetRef<Expr>(op));
    }
    return Combine<T>(analyzer_, a, b);
  }

  // recursive depth
  int recur_depth_{0};
  // analyzer
  Analyzer* analyzer_;
  const Map<Var, IntSet>& dom_map_;
  bool eval_vec_{false};
};

class IntSetAnalyzer::Impl {
 public:
  explicit Impl(Analyzer* analyzer)
      : analyzer_(analyzer) {
  }

  IntSet Eval(const Expr& expr, const Map<Var, IntSet>& dom_map) const {
    return IntervalSetEvaluator(analyzer_, dom_map).Eval(expr);
  }

 private:
  Analyzer* analyzer_;
};

IntSetAnalyzer::IntSetAnalyzer(Analyzer* parent)
    : impl_(new Impl(parent)) {
}

IntSetAnalyzer::~IntSetAnalyzer() {
  delete impl_;
}

IntSet IntSetAnalyzer::operator()(const Expr& expr,
                                  const Map<Var, IntSet>& dom_map) {
  return impl_->Eval(expr, dom_map);
}

// Quickly adapt to IntSet interface
// TODO(tqchen): revisit IntSet interface as well.
Range IntSet::cover_range(Range max_range) const {
  IntSet temp;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  CHECK(s_int != nullptr);
  if (s_int->HasUpperBound() && s_int->HasLowerBound()) {
    return Range::make_by_min_extent(
        s_int->min_value, Simplify(s_int->max_value + 1 - s_int->min_value));
  }
  return max_range;
}

Expr IntSet::min() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  CHECK(s_int);
  return s_int->min_value;
}

Expr IntSet::max() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  CHECK(s_int);
  return s_int->max_value;
}

bool IntSet::is_nothing() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsEmpty());
}

bool IntSet::is_everything() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsEverything());
}

bool IntSet::is_single_point() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsSinglePoint());
}

bool IntSet::can_prove_positive() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && is_positive_const(ir::Simplify(s_int->min_value)));
}

bool IntSet::can_prove_negative() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && is_negative_const(ir::Simplify(s_int->max_value)));
}

bool IntSet::can_prove_non_positive() const {
  if (const auto* s_int = (*this).as<IntervalSetNode>()) {
    auto max = ir::Simplify(s_int->max_value);
    return is_zero(max) || is_negative_const(max);
  }
  return false;
}

bool IntSet::can_prove_non_negative() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    auto min = ir::Simplify(s_int->min_value);
    return is_zero(min) || is_positive_const(min);
  }
  return false;
}

SignType IntSet::sign_type() const {
  if (can_prove_positive()) {
    return kPositive;
  } else if (can_prove_negative()) {
    return kNegative;
  } else if (is_single_point() && is_zero(point_value())) {
    return kZero;
  } else {
    return kUnknown;
  }
}
Expr IntSet::point_value() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  CHECK(s_int && s_int->IsSinglePoint());
  return s_int->min_value;
}

IntSet IntSet::nothing() {
  return IntervalSet::Empty();
}

IntSet IntSet::everything() {
  return IntervalSet::Everything();
}

IntSet IntSet::single_point(Expr x) {
  return IntervalSet::SinglePoint(x);
}

IntSet IntSet::interval(Expr min, Expr max) {
  if (min.same_as(max)) {
    return IntSet::single_point(min);
  }
  return IntervalSet(min, max);
}

// Range related code
inline bool ProveEqual(Expr lhs, Expr rhs) {
  return is_zero(ir::Simplify(lhs - rhs));
}

IntSet IntSet::range(Range r) {
  // must make sure it can be matched back by MatchRange.
  if (is_one(r->extent)) {
    return IntSet::single_point(r->min);
  }
  return IntervalSet(r->min, r->extent + r->min - 1);
}

bool IntSet::match_range(const Range& b) const {
  const IntSet& a = *this;
  const IntervalSetNode* a_int = a.as<IntervalSetNode>();
  if (!a_int) return false;
  return ProveEqual(a_int->min_value, b->min) &&
      ProveEqual(a_int->max_value, b->extent + b->min - 1);
}

IntSet Union(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer ana;
  IntervalSet x = ToIntervalSet(sets[0]);
  for (size_t i = 1; i < sets.size(); ++i) {
    x = Union(&ana, x, ToIntervalSet(sets[i]));
  }
  return IntervalSet(ir::Simplify(x->min_value),
                     ir::Simplify(x->max_value));
}

IntSet Intersect(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer ana;
  IntervalSet x = ToIntervalSet(sets[0]);
  for (size_t i = 1; i < sets.size(); ++i) {
    x = Intersect(&ana, x, ToIntervalSet(sets[i]));
  }
  return IntervalSet(ir::Simplify(x->min_value),
                     ir::Simplify(x->max_value));
}

Map<Var, IntSet> ConvertDomMap(const Map<IterVar, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(kv.first->var, kv.second);
  }
  return dmap;
}

Map<Var, IntSet> ConvertDomMap(
    const std::unordered_map<const Variable*, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(GetRef<Var>(kv.first), kv.second);
  }
  return dmap;
}

IntSet EvalSet(Expr e,
               const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  return IntervalSetEvaluator(&ana, dom_map, false).Eval(e);
}

IntSet IntSet::vector(Expr x) {
  Analyzer ana;
  Map<Var, IntSet> dmap;
  return IntervalSetEvaluator(&ana, dmap, true).Eval(x);
}

IntSet EvalSet(Expr e,
               const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(Expr e,
               const std::unordered_map<const Variable*, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(Range r,
               const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  IntervalSetEvaluator m(&ana, dom_map);
  // Simplifying first can give tighter bounds if r->min and r->extent share variables
  Expr sum = r->min + r->extent - 1;
  auto res  = m.Eval(IntervalSet(r->min,  Simplify(sum)));
  return std::move(res);
}

IntSet EvalSet(Range r,
               const std::unordered_map<const Variable*, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

IntSet EvalSet(IntSet s,
               const std::unordered_map<const Variable*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  IntervalSetEvaluator m(&ana, dmap);
  const IntervalSetNode* s_int = s.as<IntervalSetNode>();
  Expr vmax = s_int->HasUpperBound() ?
      m.Eval(s_int->max_value).max() : s_int->max_value;
  Expr vmin = s_int->HasLowerBound() ?
      m.Eval(s_int->min_value).min() : s_int->min_value;
  return IntervalSet(vmin, vmax);
}

class SubExprIntervalSetEvaluator : public IntervalSetEvaluator {
 public:
  explicit SubExprIntervalSetEvaluator(
      Analyzer* analyzer,
      const Map<Var, IntSet>& dom_map)
      : IntervalSetEvaluator(analyzer, dom_map) {}

  IntervalSet VisitExpr(const Expr& n) final {
    IntervalSet ret = IntervalSetEvaluator::VisitExpr(n);
    expr_map[n] = ret;
    return ret;
  }

  ExprIntSetMap expr_map;
};

ExprIntSetMap EvalSetForEachSubExpr(
    Expr e,
    const std::unordered_map<const Variable*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  SubExprIntervalSetEvaluator m(&ana, dmap);
  m.Eval(e);
  return m.expr_map;
}

IntSet EvalSet(Range r,
               const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IntervalSetNode>([](const IntervalSetNode *op, IRPrinter *p) {
    p->stream << "IntervalSet"
              << "[" << op->min_value << ", "
              << op->max_value << ']';
  });
}  // namespace arith
}  // namespace tvm
