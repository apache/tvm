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
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "constraint_extract.h"
#include "interval_set.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using tir::is_one;
using tir::is_zero;
using tir::make_const;
using tir::make_zero;

PrimExpr SymbolicLimits::pos_inf_ = Var("pos_inf", DataType::Handle());
PrimExpr SymbolicLimits::neg_inf_ = Var("neg_inf", DataType::Handle());

IntervalSet::IntervalSet(PrimExpr min_value, PrimExpr max_value) {
  auto node = make_object<IntervalSetNode>();
  node->min_value = std::move(min_value);
  node->max_value = std::move(max_value);
  data_ = std::move(node);
}

IntervalSet MakeIntervalSet(PrimExpr min_value, PrimExpr max_value) {
  return IntervalSet(min_value, max_value);
}

TVM_REGISTER_GLOBAL("arith.IntervalSet").set_body_typed(MakeIntervalSet);

IntervalSet Intersect(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  PrimExpr max_value = min(a->max_value, b->max_value);
  PrimExpr min_value = max(a->min_value, b->min_value);
  if ((max_value.dtype().is_int() || max_value.dtype().is_uint()) &&
      (min_value.dtype().is_int() || min_value.dtype().is_uint()) &&
      analyzer->CanProve(max_value < min_value)) {
    return IntervalSet::Empty();
  } else {
    return IntervalSet(min_value, max_value);
  }
}

IntervalSet Union(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsEmpty()) return b;
  if (b->IsEmpty()) return a;
  PrimExpr max_value = max(a->max_value, b->max_value);
  PrimExpr min_value = min(a->min_value, b->min_value);
  return IntervalSet(min_value, max_value);
}

// type traits
template <typename OP>
struct is_logical_op {
  static const bool value = false;
};

#define TVM_DECLARE_LOGICAL_OP(OP)  \
  template <>                       \
  struct is_logical_op<tir::OP> {   \
    static const bool value = true; \
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
template <typename Op>
inline IntervalSet Combine(Analyzer* analyzer, IntervalSet a, IntervalSet b, DataType dtype) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    PrimExpr expr;
    if (auto res = TryConstFold<Op>(a->min_value, b->min_value)) {
      expr = res.value();
    } else {
      expr = Op(a->min_value, b->min_value);
    }
    return IntervalSet::SinglePoint(expr);
  }
  if (is_logical_op<Op>::value) {
    return IntervalSet(make_const(dtype, 0), make_const(dtype, 1));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (a->IsEverything()) return a;
  if (b->IsEverything()) return b;
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Add>(Analyzer* analyer, IntervalSet a, IntervalSet b,
                                     DataType /* dtype */) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value + b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  PrimExpr min_value =
      a->HasLowerBound() && b->HasLowerBound() ? a->min_value + b->min_value : neg_inf();
  PrimExpr max_value =
      a->HasUpperBound() && b->HasUpperBound() ? a->max_value + b->max_value : pos_inf();
  return IntervalSet(min_value, max_value);
}

template <>
inline IntervalSet Combine<tir::Sub>(Analyzer* analyer, IntervalSet a, IntervalSet b,
                                     DataType /* dtype */) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value - b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  PrimExpr min_value =
      a->HasLowerBound() && b->HasUpperBound() ? a->min_value - b->max_value : neg_inf();
  PrimExpr max_value =
      a->HasUpperBound() && b->HasLowerBound() ? a->max_value - b->min_value : pos_inf();
  return IntervalSet(min_value, max_value);
}

template <>
inline IntervalSet Combine<tir::Mul>(Analyzer* analyzer, IntervalSet a, IntervalSet b,
                                     DataType /* dtype */) {
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
      PrimExpr min_value = a->HasLowerBound() ? a->min_value * b->min_value : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? a->max_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? a->max_value * b->min_value : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? a->min_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::Select;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = a->min_value * b->min_value;
      PrimExpr e2 = a->max_value * b->min_value;
      return IntervalSet(Select(sign, e1, e2), Select(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mul";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Div>(Analyzer* analyzer, IntervalSet a, IntervalSet b,
                                     DataType /* dtype */) {
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
      PrimExpr min_value = a->HasLowerBound() ? a->min_value / b->min_value : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? a->max_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? a->max_value / b->min_value : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? a->min_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::Select;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = a->min_value / b->min_value;
      PrimExpr e2 = a->max_value / b->min_value;
      return IntervalSet(Select(sign, e1, e2), Select(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Mod>(Analyzer* analyzer, IntervalSet a, IntervalSet b,
                                     DataType /* dtype */) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(truncmod(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const PrimExpr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    // We need to add more bound constraints throughout the code.
    // The logic below assumes a is non-negative, which usually
    // is the case of our application.
    // TODO(tqchen): add bound constraints for a.
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      return IntervalSet(make_zero(divisor.dtype()), divisor - 1);
    } else {
      PrimExpr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::FloorDiv>(Analyzer* analyzer, IntervalSet a, IntervalSet b,
                                          DataType /* dtype */) {
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
      PrimExpr min_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::Select;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = floordiv(a->min_value, b->min_value);
      PrimExpr e2 = floordiv(a->max_value, b->min_value);
      return IntervalSet(Select(sign, e1, e2), Select(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::FloorMod>(Analyzer* analyzer, IntervalSet a, IntervalSet b,
                                          DataType /* dtype */) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(floormod(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const PrimExpr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      if (divisor.as<tir::IntImmNode>()) {
        // a mod b = a - (a / b) * b if a_max / b == a_min / b
        auto qmax = a->HasUpperBound() ? floordiv(a->max_value, divisor) : pos_inf();
        auto qmin = a->HasLowerBound() ? floordiv(a->min_value, divisor) : neg_inf();
        // We can compare +/- inf against each other, but cannot use
        // operator== between the symbolic limits and an integer.
        bool compatible_dtypes = !(qmin.dtype().is_handle() ^ qmax.dtype().is_handle());
        if (compatible_dtypes && analyzer->CanProve(qmax == qmin)) {
          auto tmax = a->max_value - divisor * qmin;
          auto tmin = a->min_value - divisor * qmin;
          return IntervalSet(tmin, tmax);
        }
      }
      return IntervalSet(make_zero(divisor.dtype()), divisor - 1);
    } else {
      PrimExpr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Max>(Analyzer* analzyer, IntervalSet a, IntervalSet b,
                                     DataType /* dtype */) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(max(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(max(a->min_value, b->min_value), max(a->max_value, b->max_value));
}

template <>
inline IntervalSet Combine<tir::Min>(Analyzer* analzyer, IntervalSet a, IntervalSet b,
                                     DataType /* dtype */) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(min(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(min(a->min_value, b->min_value), min(a->max_value, b->max_value));
}

// internal helper function to get an interval set
IntervalSet ToIntervalSet(IntSet set) {
  if (auto node = set.as<IntervalSet>()) {
    return node.value();
  }
  DLOG(INFO) << "cannot resolve int set " << set;
  return IntervalSet::Everything();
}

using namespace tir;

// Simplified version of int set evaluator that operates on IntervalSet
// We might use better set analysis in the future to replace the intervalset.
class IntervalSetEvaluator : public ExprFunctor<IntervalSet(const PrimExpr&)> {
 public:
  IntervalSetEvaluator(Analyzer* analyzer, const Map<Var, IntSet>& dom_map,
                       const std::vector<std::pair<Var, IntSet>>* dom_constraints = nullptr,
                       bool eval_vec = false)
      : analyzer_(analyzer),
        dom_map_(dom_map),
        dom_constraints_(dom_constraints),
        eval_vec_(eval_vec) {}

  IntervalSet Eval(const PrimExpr& val) { return this->VisitExpr(val); }
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

  IntervalSet VisitExpr_(const IntImmNode* op) final {
    return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
  }

  IntervalSet VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    Array<IntSet> values;
    if (dom_constraints_) {
      for (const auto& constraint : *dom_constraints_) {
        if (var.same_as(constraint.first)) {
          values.push_back(constraint.second);
        }
      }
    }

    auto it = dom_map_.find(var);
    if (it != dom_map_.end()) {
      values.push_back((*it).second);
    }

    if (values.empty()) {
      return IntervalSet::SinglePoint(var);
    }

    IntSet intersection = [&]() {
      if (values.size() == 1) {
        return values.front();
      } else {
        return Intersect(values);
      }
    }();

    IntervalSet res = ToIntervalSet(intersection);
    if (res->min_value.same_as(var) && res->max_value.same_as(var)) {
      return res;
    }
    // recursively evaluate mapped result
    // in case the domain contains variables to be relaxed.
    return Eval(res);
  }

  IntervalSet VisitExpr_(const AddNode* op) final { return VisitBinaryExpr_<Add>(op); }

  IntervalSet VisitExpr_(const SubNode* op) final { return VisitBinaryExpr_<Sub>(op); }

  IntervalSet VisitExpr_(const MulNode* op) final { return VisitBinaryExpr_<Mul>(op); }

  IntervalSet VisitExpr_(const DivNode* op) final { return VisitBinaryExpr_<Div>(op); }

  IntervalSet VisitExpr_(const ModNode* op) final { return VisitBinaryExpr_<Mod>(op); }

  IntervalSet VisitExpr_(const FloorDivNode* op) final { return VisitBinaryExpr_<FloorDiv>(op); }

  IntervalSet VisitExpr_(const FloorModNode* op) final { return VisitBinaryExpr_<FloorMod>(op); }

  IntervalSet VisitExpr_(const MinNode* op) final { return VisitBinaryExpr_<Min>(op); }

  IntervalSet VisitExpr_(const MaxNode* op) final { return VisitBinaryExpr_<Max>(op); }

  IntervalSet VisitExpr_(const EQNode* op) final { return VisitBinaryExpr_<EQ>(op); }

  IntervalSet VisitExpr_(const NENode* op) final { return VisitBinaryExpr_<NE>(op); }

  IntervalSet VisitExpr_(const LTNode* op) final { return VisitBinaryExpr_<LT>(op); }

  IntervalSet VisitExpr_(const LENode* op) final { return VisitBinaryExpr_<LE>(op); }

  IntervalSet VisitExpr_(const GTNode* op) final { return VisitBinaryExpr_<GT>(op); }

  IntervalSet VisitExpr_(const GENode* op) final { return VisitBinaryExpr_<GE>(op); }

  IntervalSet VisitExpr_(const AndNode* op) final { return VisitBinaryExpr_<And>(op); }

  IntervalSet VisitExpr_(const OrNode* op) final { return VisitBinaryExpr_<Or>(op); }

  IntervalSet VisitExpr_(const RampNode* op) final {
    ICHECK(eval_vec_);
    IntervalSet base = Eval(op->base);
    PVar<IntImm> stride;
    if (stride.Match(op->stride)) {
      DataType t = op->base.dtype();
      int64_t vstride = stride.Eval()->value;
      if (vstride > 0) {
        return Combine<Add>(analyzer_, base,
                            IntervalSet(make_zero(t), make_const(t, vstride * (op->lanes - 1))),
                            op->dtype);
      } else {
        return Combine<Add>(analyzer_, base,
                            IntervalSet(make_const(t, vstride * (op->lanes - 1)), make_zero(t)),
                            op->dtype);
      }
    }
    DLOG(WARNING) << "cannot evaluate set on expression " << GetRef<PrimExpr>(op);
    return IntervalSet::Everything();
  }

  IntervalSet VisitExpr_(const BroadcastNode* op) final {
    ICHECK(eval_vec_);
    return VisitExpr(op->value);
  }

  IntervalSet VisitExpr_(const SelectNode* op) final {
    IntervalSet true_set = this->Eval(op->true_value);
    IntervalSet false_set = this->Eval(op->false_value);
    return Union(analyzer_, false_set, true_set);
  }

  IntervalSet VisitExpr_(const CastNode* op) final {
    IntervalSet value_set = this->Eval(op->value);
    // short cut for the int set.
    if (value_set->min_value.same_as(value_set->max_value)) {
      if (value_set->IsEmpty()) return value_set;
      return IntervalSet::SinglePoint(cast(op->dtype, value_set->min_value));
    }
    PrimExpr min_value =
        value_set->HasLowerBound() ? cast(op->dtype, value_set->min_value) : neg_inf();
    PrimExpr max_value =
        value_set->HasUpperBound() ? cast(op->dtype, value_set->max_value) : pos_inf();
    return IntervalSet(min_value, max_value);
  }

  IntervalSet VisitExpr_(const BufferLoadNode* op) final {
    if (!(op->dtype.is_int() || op->dtype.is_uint())) {
      DLOG(WARNING) << "cannot evaluate set BufferLoad which loads from a " << op->dtype
                    << " buffer";
      return IntervalSet::Everything();
    }
    // If the indices do not contain any variables to be relaxed, return the BufferLoad itself.
    // Otherwise return `IntervalSet::everything()` since we have no knowledge on the buffer data.
    for (const PrimExpr& index : op->indices) {
      if (UsesVar(index, [dom_map = &this->dom_map_](const VarNode* var) {
            return dom_map->find(GetRef<Var>(var)) != dom_map->end();
          })) {
        return IntervalSet::Everything();
      }
    }
    return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
  }

  IntervalSet VisitExprDefault_(const Object* op) final {
    DLOG(WARNING) << "cannot evaluate set type " << op->GetTypeKey();
    return IntervalSet::Everything();
  }

 private:
  // whether set is exactly single point that equals value.
  bool MatchPoint(const IntervalSet& set, const PrimExpr& value) const {
    return set->min_value.same_as(value) && set->max_value.same_as(value);
  }

  template <typename TOp, typename T>
  inline IntervalSet VisitBinaryExpr_(const T* op) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    IntervalSet a = this->Eval(op->a);
    IntervalSet b = this->Eval(op->b);
    if (MatchPoint(a, op->a) && MatchPoint(b, op->b)) {
      return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
    }
    return Combine<TOp>(analyzer_, a, b, op->dtype);
  }

  // recursive depth
  int recur_depth_{0};
  // analyzer
  Analyzer* analyzer_;
  const Map<Var, IntSet>& dom_map_;
  const std::vector<std::pair<Var, IntSet>>* dom_constraints_;
  bool eval_vec_{false};
};

class IntSetAnalyzer::Impl {
 public:
  explicit Impl(Analyzer* analyzer) : analyzer_(analyzer) {}

  IntSet Eval(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) const {
    return IntervalSetEvaluator(analyzer_, dom_map).Eval(expr);
  }

  IntSet Eval(const PrimExpr& expr) const {
    return IntervalSetEvaluator(analyzer_, dom_map_, &dom_constraints_, true).Eval(expr);
  }

  void Bind(const Var& var, const Range& range, bool allow_override) {
    Update(var, IntSet::FromRange(range), allow_override);
  }

  void Update(const Var& var, const IntSet& info, bool override_info);
  void Bind(const Var& var, const PrimExpr& expr, bool override_info);
  std::function<void()> EnterConstraint(const PrimExpr& constraint);

 private:
  // Utility function to split a boolean condition into the domain
  // bounds implied by that condition.
  static std::vector<std::pair<Var, IntSet>> DetectBoundInfo(const PrimExpr& cond);

  // The parent arith::Analyzer
  Analyzer* analyzer_;

  // Map of variables to global variable bounds (e.g. loop iterator
  // ranges)
  Map<Var, IntSet> dom_map_;

  // List of implicit scope-dependent bounds (e.g. inside the body of
  // an if-statement).  Maintained as a list of constraints, rather
  // than as a `Map<Var,IntSet>`, to avoid computing an Intersection
  // until required.
  std::vector<std::pair<Var, IntSet>> dom_constraints_;
};

IntSetAnalyzer::IntSetAnalyzer(Analyzer* parent) : impl_(new Impl(parent)) {}

IntSetAnalyzer::~IntSetAnalyzer() { delete impl_; }

IntSet IntSetAnalyzer::operator()(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) {
  return impl_->Eval(expr, dom_map);
}

IntSet IntSetAnalyzer::operator()(const PrimExpr& expr) { return impl_->Eval(expr); }

void IntSetAnalyzer::Update(const Var& var, const IntSet& info, bool allow_override) {
  impl_->Update(var, info, allow_override);
}

void IntSetAnalyzer::Bind(const Var& var, const Range& range, bool allow_override) {
  impl_->Bind(var, range, allow_override);
}

void IntSetAnalyzer::Impl::Update(const Var& var, const IntSet& info, bool can_override) {
  if (!can_override) {
    auto it = dom_map_.find(var);
    if (it != dom_map_.end()) {
      const IntSet& old_info = (*it).second;

      ICHECK(ExprDeepEqual()(old_info.min(), info.min()))
          << "Trying to update var \'" << var << "\'"
          << " with a different minimum value: "
          << "original=" << old_info.min() << ", new=" << info.min();

      ICHECK(ExprDeepEqual()(old_info.max(), info.max()))
          << "Trying to update var \'" << var << "\'"
          << " with a different maximum value: "
          << "original=" << old_info.max() << ", new=" << info.max();
    }
  }
  dom_map_.Set(var, info);
}

void IntSetAnalyzer::Impl::Bind(const Var& var, const PrimExpr& expr, bool can_override) {
  Update(var, Eval(expr), can_override);
}

std::vector<std::pair<Var, IntSet>> IntSetAnalyzer::Impl::DetectBoundInfo(
    const PrimExpr& constraint) {
  PVar<Var> x;
  PVar<PrimExpr> limit;

  std::vector<std::pair<Var, IntSet>> bounds;
  for (const PrimExpr& subconstraint : ExtractConstraints(constraint)) {
    if ((x <= limit).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(SymbolicLimits::neg_inf_, limit.Eval())});
    } else if ((x < limit).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(SymbolicLimits::neg_inf_, limit.Eval() - 1)});
    } else if ((x >= limit).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(limit.Eval(), SymbolicLimits::pos_inf_)});
    } else if ((x > limit).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(limit.Eval() + 1, SymbolicLimits::pos_inf_)});
    } else if ((x == limit).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::SinglePoint(limit.Eval())});
    }

    if ((limit >= x).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(SymbolicLimits::neg_inf_, limit.Eval())});
    } else if ((limit > x).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(SymbolicLimits::neg_inf_, limit.Eval() - 1)});
    } else if ((limit <= x).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(limit.Eval(), SymbolicLimits::pos_inf_)});
    } else if ((limit < x).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::Interval(limit.Eval() + 1, SymbolicLimits::pos_inf_)});
    } else if ((limit == x).Match(subconstraint)) {
      bounds.push_back({x.Eval(), IntSet::SinglePoint(limit.Eval())});
    }
  }
  return bounds;
}

std::function<void()> IntSetAnalyzer::EnterConstraint(const PrimExpr& constraint) {
  return impl_->EnterConstraint(constraint);
}

std::function<void()> IntSetAnalyzer::Impl::EnterConstraint(const PrimExpr& constraint) {
  auto bounds = DetectBoundInfo(constraint);

  if (bounds.size() == 0) return nullptr;

  size_t old_size = dom_constraints_.size();
  dom_constraints_.insert(dom_constraints_.end(), bounds.begin(), bounds.end());
  size_t new_size = dom_constraints_.size();
  auto frecover = [old_size, new_size, this]() {
    ICHECK_EQ(dom_constraints_.size(), new_size);
    dom_constraints_.resize(old_size);
  };
  return frecover;
}

// Quickly adapt to IntSet interface
// TODO(tqchen): revisit IntSet interface as well.
Range IntSet::CoverRange(Range max_range) const {
  IntSet temp;
  Analyzer analyzer;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int != nullptr);
  if (s_int->HasUpperBound() && s_int->HasLowerBound()) {
    return Range::FromMinExtent(analyzer.Simplify(s_int->min_value),
                                analyzer.Simplify(s_int->max_value + 1 - s_int->min_value));
  }
  return max_range;
}

PrimExpr IntSet::min() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int);
  return s_int->min_value;
}

PrimExpr IntSet::max() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int);
  return s_int->max_value;
}

bool IntSet::IsNothing() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsEmpty());
}

bool IntSet::IsEverything() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsEverything());
}

bool IntSet::IsSinglePoint() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsSinglePoint());
}

bool IntSet::CanProveSinglePoint(Analyzer* ana) const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  if (!s_int) return false;
  if (s_int->IsSinglePoint()) return true;
  return ana->CanProveEqual(s_int->min_value, s_int->max_value);
}

bool IntSet::CanProvePositive() const {
  Analyzer analyzer;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && is_positive_const(analyzer.Simplify(s_int->min_value)));
}

bool IntSet::CanProveNegative() const {
  Analyzer analyzer;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && is_negative_const(analyzer.Simplify(s_int->max_value)));
}

bool IntSet::CanProveNonPositive() const {
  Analyzer analyzer;
  if (const auto* s_int = (*this).as<IntervalSetNode>()) {
    auto max = analyzer.Simplify(s_int->max_value);
    return is_zero(max) || is_negative_const(max);
  }
  return false;
}

bool IntSet::CanProveNonNegative() const {
  Analyzer analyzer;
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    auto min = analyzer.Simplify(s_int->min_value);
    return is_zero(min) || is_positive_const(min);
  }
  return false;
}

bool IntSet::HasLowerBound() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    return s_int->HasLowerBound();
  }
  return false;
}

bool IntSet::HasUpperBound() const {
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    return s_int->HasUpperBound();
  }
  return false;
}

SignType IntSet::GetSignType() const {
  if (CanProvePositive()) {
    return kPositive;
  } else if (CanProveNegative()) {
    return kNegative;
  } else if (IsSinglePoint() && is_zero(PointValue())) {
    return kZero;
  } else {
    return kUnknown;
  }
}
PrimExpr IntSet::PointValue() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int && s_int->IsSinglePoint());
  return s_int->min_value;
}

IntSet IntSet::Nothing() { return IntervalSet::Empty(); }

IntSet IntSet::Everything() { return IntervalSet::Everything(); }

IntSet IntSet::SinglePoint(PrimExpr x) { return IntervalSet::SinglePoint(x); }

IntSet IntSet::Interval(PrimExpr min, PrimExpr max) {
  if (min.same_as(max)) {
    return IntSet::SinglePoint(min);
  }
  return IntervalSet(min, max);
}

// Range related code
inline bool ProveEqual(Analyzer* analyzer, PrimExpr lhs, PrimExpr rhs) {
  return is_zero(analyzer->Simplify(lhs - rhs));
}

IntSet IntSet::FromMinExtent(PrimExpr min, PrimExpr extent) {
  if (is_one(extent)) {
    return IntSet::SinglePoint(min);
  }
  return IntervalSet(min, extent + min - 1);
}

IntSet IntSet::FromRange(Range r) {
  // must make sure it can be matched back by MatchRange.
  if (is_one(r->extent)) {
    return IntSet::SinglePoint(r->min);
  }
  return IntervalSet(r->min, r->extent + r->min - 1);
}

bool IntSet::MatchRange(const Range& b) const {
  const IntSet& a = *this;
  const IntervalSetNode* a_int = a.as<IntervalSetNode>();
  if (!a_int) return false;
  if (!a_int->HasUpperBound() || !a_int->HasLowerBound()) return false;
  Analyzer ana;
  return ProveEqual(&ana, a_int->min_value, b->min) &&
         ProveEqual(&ana, a_int->max_value, b->extent + b->min - 1);
}

IntSet Union(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::Nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer ana;
  IntervalSet x = ToIntervalSet(sets[0]);
  for (size_t i = 1; i < sets.size(); ++i) {
    x = Union(&ana, x, ToIntervalSet(sets[i]));
  }
  return IntervalSet(ana.Simplify(x->min_value), ana.Simplify(x->max_value));
}

Array<IntSet> UnionRegion(const Array<Array<IntSet>>& nd_int_sets) {
  if (nd_int_sets.empty()) {
    return {};
  }
  int n = nd_int_sets.size();
  int ndim = nd_int_sets[0].size();
  Array<IntSet> result;
  result.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    Array<IntSet> candidates;
    candidates.reserve(n);
    for (int j = 0; j < n; ++j) {
      candidates.push_back(nd_int_sets[j][i]);
    }
    result.push_back(Union(candidates));
  }
  return result;
}

IntSet UnionLowerBound(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::Nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer analyzer;
  bool is_first_interval = true;
  PrimExpr min_inclusive{nullptr};
  PrimExpr max_inclusive(nullptr);
  for (const IntSet& int_set : sets) {
    if (int_set.IsNothing()) continue;
    if (const auto* interval_set = int_set.as<IntervalSetNode>()) {
      PrimExpr new_min_inclusive = interval_set->min_value;
      PrimExpr new_max_inclusive = interval_set->max_value;
      if (is_first_interval) {
        is_first_interval = false;
        min_inclusive = std::move(new_min_inclusive);
        max_inclusive = std::move(new_max_inclusive);
        continue;
      }
      bool bound_1 = is_neg_inf(new_min_inclusive) || is_pos_inf(max_inclusive) ||
                     analyzer.CanProve(new_min_inclusive <= max_inclusive + 1);
      bool bound_2 = is_neg_inf(min_inclusive) || is_pos_inf(new_max_inclusive) ||
                     analyzer.CanProve(min_inclusive <= new_max_inclusive + 1);
      if (bound_1 && bound_2) {
        min_inclusive = min(min_inclusive, new_min_inclusive);
        max_inclusive = max(max_inclusive, new_max_inclusive);
      }
    }
  }
  if (is_first_interval) {
    return IntSet::Nothing();
  }
  return IntSet::Interval(min_inclusive, max_inclusive);
}

Array<IntSet> UnionRegionLowerBound(const Array<Array<IntSet>>& nd_int_sets) {
  if (nd_int_sets.empty()) {
    return {};
  }
  int n = nd_int_sets.size();
  int ndim = nd_int_sets[0].size();
  Array<IntSet> result;
  result.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    Array<IntSet> candidates;
    candidates.reserve(n);
    for (int j = 0; j < n; ++j) {
      candidates.push_back(nd_int_sets[j][i]);
    }
    result.push_back(UnionLowerBound(candidates));
  }
  return result;
}

IntSet Intersect(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::Nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer ana;
  IntervalSet x = ToIntervalSet(sets[0]);
  for (size_t i = 1; i < sets.size(); ++i) {
    x = Intersect(&ana, x, ToIntervalSet(sets[i]));
  }
  return IntervalSet(ana.Simplify(x->min_value), ana.Simplify(x->max_value));
}

Map<Var, IntSet> ConvertDomMap(const Map<IterVar, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(kv.first->var, kv.second);
  }
  return dmap;
}

Map<Var, IntSet> ConvertDomMap(const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(GetRef<Var>(kv.first), kv.second);
  }
  return dmap;
}

IntSet EvalSet(PrimExpr e, const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  return IntervalSetEvaluator(&ana, dom_map, {}, false).Eval(e);
}

IntSet IntSet::Vector(PrimExpr x) {
  // short cut: simply get single point
  if (x.dtype().lanes() == 1) {
    return IntSet::SinglePoint(x);
  } else {
    // vector case.
    Analyzer ana;
    Map<Var, IntSet> dmap;
    return IntervalSetEvaluator(&ana, dmap, {}, true).Eval(x);
  }
}

IntSet EvalSet(PrimExpr e, const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(PrimExpr e, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(Range r, const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  if ((r->min->dtype.is_int() || r->min->dtype.is_uint()) && ana.CanProveEqual(r->extent, 1)) {
    return EvalSet(r->min, dom_map);
  }
  IntervalSetEvaluator m(&ana, dom_map);
  // Simplifying first can give tighter bounds if r->min and r->extent share variables
  PrimExpr sum = r->min + r->extent - 1;
  auto res = m.Eval(IntervalSet(r->min, ana.Simplify(sum)));
  return std::move(res);
}

IntSet EvalSet(Range r, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

Array<IntSet> EvalSet(const Array<Range>& region, const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  IntervalSetEvaluator m(&ana, dom_map);
  Array<IntSet> result;
  result.reserve(region.size());
  for (const Range& r : region) {
    PrimExpr sum = r->min + (r->extent - 1);
    result.push_back(m.Eval(IntervalSet(r->min, ana.Simplify(sum))));
  }
  return result;
}

IntSet EvalSet(IntSet s, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  IntervalSetEvaluator m(&ana, dmap);
  const IntervalSetNode* s_int = s.as<IntervalSetNode>();
  PrimExpr vmax = s_int->HasUpperBound() ? m.Eval(s_int->max_value).max() : s_int->max_value;
  PrimExpr vmin = s_int->HasLowerBound() ? m.Eval(s_int->min_value).min() : s_int->min_value;
  return IntervalSet(vmin, vmax);
}

class SubExprIntervalSetEvaluator : public IntervalSetEvaluator {
 public:
  explicit SubExprIntervalSetEvaluator(Analyzer* analyzer, const Map<Var, IntSet>& dom_map)
      : IntervalSetEvaluator(analyzer, dom_map) {}

  IntervalSet VisitExpr(const PrimExpr& n) final {
    IntervalSet ret = IntervalSetEvaluator::VisitExpr(n);
    expr_map[n] = ret;
    return ret;
  }

  ExprIntSetMap expr_map;
};

ExprIntSetMap EvalSetForEachSubExpr(PrimExpr e,
                                    const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  SubExprIntervalSetEvaluator m(&ana, dmap);
  m.Eval(e);
  return m.expr_map;
}

IntSet EvalSet(Range r, const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

Map<Var, arith::IntSet> AsIntSet(const Map<Var, Range>& var_dom) {
  Map<Var, arith::IntSet> result;
  for (auto kv : var_dom) {
    const Var& var = kv.first;
    const Range& range = kv.second;
    result.Set(var, arith::IntSet::FromRange(range));
  }
  return result;
}

/*! \brief Helper function to convert IterSumExpr to the actual touched range. */
static Optional<IntSet> EvalIterSum(const IterSumExpr& iter_min, const PrimExpr& extent,
                                    Analyzer* analyzer) {
  if (analyzer->CanProve(extent == 0)) {
    return IntSet::Nothing();
  }
  if (iter_min->args.empty()) {
    return IntSet::FromMinExtent(iter_min->base, extent);
  }
  ICHECK_EQ(iter_min->args.size(), 1) << "The `EvalIterSum` expects fused iter sum expr";
  const IterSplitExpr& split = iter_min->args[0];
  if (analyzer->CanProve(split->extent == 0)) {
    return IntSet::Nothing();
  }
  if (!analyzer->CanProve(extent >= split->scale)) {
    return NullOpt;
  }

  const PrimExpr& base = iter_min->base;
  // IterSplitExpr: (source // lower_factor) % extent * scale
  // where `(source // lower_factor) % extent` is within [0, extent - 1]
  if (analyzer->CanProve(split->scale < 0)) {
    // If scale is negative, the var dom is [(extent - 1) * scale, 0]
    // The total base is `base + (extent - 1) * scale`,
    // while total extent is `dom_extent + (extent - 1) * (-scale)`
    const PrimExpr& var_extent = (split->extent - 1) * split->scale;
    return IntSet::FromMinExtent(base + var_extent, extent - var_extent);
  } else {
    // If scale is positive, the var dom is [0, (extent - 1) * scale]
    // The total dom is [base, dom_extent + (extent - 1) * scale]
    return IntSet::FromMinExtent(base, extent + (split->extent - 1) * split->scale);
  }
}

Optional<Array<IntSet>> EstimateRegionStrictBound(const Array<Range>& region,
                                                  const Map<Var, Range>& var_dom,
                                                  const PrimExpr& predicate, Analyzer* analyzer) {
  int ndim = region.size();
  Array<IterSumExpr> iter_sum_exprs{nullptr};
  {
    Array<PrimExpr> affine_indices;
    affine_indices.reserve(ndim);
    for (const Range& range : region) {
      if (!is_const_number(range->extent)) {
        // dynamic extent is not supported yet.
        return NullOpt;
      }
      affine_indices.push_back(range->min);
    }
    auto res = DetectIterMap(
        /*indices=*/affine_indices, /*input_iters=*/var_dom,
        /*predicate=*/predicate, /*check_level=*/IterMapLevel::Surjective, analyzer);
    iter_sum_exprs = res->indices;
  }
  if (iter_sum_exprs.empty()) {
    return NullOpt;
  }
  ICHECK_EQ(iter_sum_exprs.size(), ndim);
  Array<IntSet> result;
  result.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    const IterSumExpr& sum_expr = iter_sum_exprs[i];
    const Range& range = region[i];
    Optional<IntSet> int_set = EvalIterSum(sum_expr, range->extent, analyzer);
    if (int_set.defined()) {
      result.push_back(int_set.value());
    } else {
      return NullOpt;
    }
  }
  return result;
}

Optional<Array<IntSet>> EstimateRegionLowerBound(const Array<Range>& region,
                                                 const Map<Var, Range>& var_dom,
                                                 const PrimExpr& predicate,
                                                 arith::Analyzer* analyzer) {
  return EstimateRegionStrictBound(region, var_dom, predicate, analyzer);
}

Array<IntSet> EstimateRegionUpperBound(const Array<Range>& region, const Map<Var, Range>& var_dom,
                                       const PrimExpr& predicate, Analyzer* analyzer) {
  if (Optional<Array<arith::IntSet>> result = EstimateRegionStrictBound(
          /*region=*/region,
          /*var_dom=*/var_dom,
          /*predicate=*/predicate, /*analyzer=*/analyzer)) {
    return result.value();
  }
  Array<IntSet> result;
  result.reserve(region.size());
  // try estimate each dimension independently
  for (const Range& range : region) {
    auto res = DetectIterMap(
        /*indices=*/{range->min}, /*input_iters=*/var_dom,
        /*predicate=*/predicate, /*check_level=*/IterMapLevel::Surjective, analyzer);
    if (!res->indices.empty()) {
      ICHECK_EQ(res->indices.size(), 1U);
      IterSumExpr sum_expr = res->indices[0];

      // dynamic extent is not supported yet.
      PrimExpr extent = range->extent;
      if (!is_const_number(extent)) {
        IntSet relaxed = EvalSet(extent, AsIntSet(var_dom));
        ICHECK(relaxed.HasUpperBound());
        extent = relaxed.max();
      }

      if (Optional<IntSet> int_set = EvalIterSum(sum_expr, range->extent, analyzer)) {
        result.push_back(int_set.value());
        continue;
      }
    }
    // fallback to coarse grained evalset
    result.push_back(EvalSet(range, AsIntSet(var_dom)));
  }
  return result;
}

TVM_REGISTER_NODE_TYPE(IntervalSetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntervalSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IntervalSetNode*>(node.get());
      p->stream << "IntervalSet"
                << "[" << op->min_value << ", " << op->max_value << ']';
    });

TVM_REGISTER_GLOBAL("arith.intset_single_point").set_body_typed(IntSet::SinglePoint);

TVM_REGISTER_GLOBAL("arith.intset_vector").set_body_typed(IntSet::Vector);

TVM_REGISTER_GLOBAL("arith.intset_interval").set_body_typed(IntSet::Interval);

TVM_REGISTER_GLOBAL("arith.IntervalSetGetMin").set_body_method(&IntSet::min);

TVM_REGISTER_GLOBAL("arith.IntervalSetGetMax").set_body_method(&IntSet::max);

TVM_REGISTER_GLOBAL("arith.IntSetIsNothing").set_body_method(&IntSet::IsNothing);

TVM_REGISTER_GLOBAL("arith.IntSetIsEverything").set_body_method(&IntSet::IsEverything);

TVM_REGISTER_GLOBAL("arith.EstimateRegionLowerBound")
    .set_body_typed([](Array<Range> region, Map<Var, Range> var_dom,
                       PrimExpr predicate) -> Optional<Array<IntSet>> {
      Analyzer analyzer;
      return EstimateRegionLowerBound(region, var_dom, predicate, &analyzer);
    });
TVM_REGISTER_GLOBAL("arith.EstimateRegionStrictBound")
    .set_body_typed([](Array<Range> region, Map<Var, Range> var_dom,
                       PrimExpr predicate) -> Optional<Array<IntSet>> {
      Analyzer analyzer;
      return EstimateRegionStrictBound(region, var_dom, predicate, &analyzer);
    });
TVM_REGISTER_GLOBAL("arith.EstimateRegionUpperBound")
    .set_body_typed([](Array<Range> region, Map<Var, Range> var_dom,
                       PrimExpr predicate) -> Optional<Array<IntSet>> {
      Analyzer analyzer;
      return EstimateRegionUpperBound(region, var_dom, predicate, &analyzer);
    });

TVM_REGISTER_GLOBAL("arith.PosInf").set_body_typed([]() { return SymbolicLimits::pos_inf_; });
TVM_REGISTER_GLOBAL("arith.NegInf").set_body_typed([]() { return SymbolicLimits::neg_inf_; });
TVM_REGISTER_GLOBAL("arith.UnionLowerBound").set_body_typed(UnionLowerBound);

}  // namespace arith
}  // namespace tvm
