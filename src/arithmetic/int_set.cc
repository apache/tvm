/*!
 *  Copyright (c) 2017 by Contributors
 * \file int_set.cc
 * \brief The integer set functions
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <tvm/ir_functor_ext.h>
#include <arithmetic/Interval.h>
#include <unordered_map>
#include "compute_expr.h"
#include "int_set_internal.h"

namespace tvm {
namespace arith {

using HalideIR::Internal::Interval;
using namespace ir;

inline IntSet IntSet::cover_interval() const {
  if ((*this).as<IntervalSet>()) return *this;
  const StrideSet* s =  (*this).as<StrideSet>();
  if (s) {
    CHECK_NE(s->extents.size(), 0U);
    Expr max = s->base.max;
    for (size_t i = 0; i < s->extents.size(); ++i) {
      max = max + s->extents[i] * s->strides[i] - s->strides[i];
    }
    return IntervalSet::make(s->base.min, Simplify(max));
  }
  LOG(FATAL) << "cannot convert set " << (*this)->type_key() << " to interval";
  return IntSet::everything();
}

Range IntSet::cover_range(Range max_range) const {
  IntSet temp;
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  if (s_int == nullptr) {
    temp = this->cover_interval();
    s_int = temp.as<IntervalSet>();
  }
  if (s_int->i.is_bounded()) {
    return Range::make_by_min_extent(
        s_int->i.min, Simplify(s_int->i.max + 1 - s_int->i.min));
  }
  return max_range;
}

Expr IntSet::min() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  CHECK(s_int);
  return s_int->i.min;
}

Expr IntSet::max() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  CHECK(s_int);
  return s_int->i.max;
}

bool IntSet::is_nothing() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  return (s_int && s_int->i.is_empty());
}

bool IntSet::is_everything() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  return (s_int && s_int->i.is_everything());
}

bool IntSet::is_single_point() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  return (s_int && s_int->i.is_single_point());
}

bool IntSet::can_prove_positive() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  return (s_int && is_positive_const(ir::Simplify(s_int->i.min)));
}

bool IntSet::can_prove_negative() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  return (s_int && is_negative_const(ir::Simplify(s_int->i.max)));
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
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  CHECK(s_int && s_int->i.is_single_point());
  return s_int->i.min;
}

IntSet IntSet::nothing() {
  return IntervalSet::make(Interval::nothing());
}

IntSet IntSet::everything() {
  return IntervalSet::make(Interval::everything());
}

IntSet IntSet::single_point(Expr x) {
  return IntervalSet::make(Interval::single_point(x));
}

IntSet IntSet::range(Range r) {
  // must make sure it can be matched back by MatchRange.
  if (is_one(r->extent)) {
    return IntSet::single_point(r->min);
  }
  if (is_positive_const(r->extent) && is_const(r->min)) {
    return IntervalSet::make(
        r->min, ComputeExpr<Sub>(ComputeExpr<Add>(r->extent, r->min), 1));
  }
  return IntervalSet::make(r->min, (r->extent + r->min) - 1);
}

IntSet IntSet::interval(Expr min, Expr max) {
  if (min.same_as(max)) {
    return IntSet::single_point(min);
  }
  return IntervalSet::make(min, max);
}

inline bool prove_equal(Expr lhs, Expr rhs) {
  return is_zero(ir::Simplify(lhs - rhs));
}

// Check if a is created from b.
bool IntSet::match_range(const Range& b) const {
  const IntSet& a = *this;
  const IntervalSet* a_int = a.as<IntervalSet>();
  if (!a_int) return false;
  const Interval& i = a_int->i;
  return prove_equal(i.min, b->min) &&
      prove_equal(i.max, ComputeExpr<Sub>(ComputeExpr<Add>(b->extent, b->min), 1));
}

inline bool MatchPoint(const IntSet& a,
                       const Expr& b) {
  const IntervalSet* a_int = a.as<IntervalSet>();
  if (!a_int) return false;
  const Interval& i = a_int->i;
  return i.is_single_point() && i.min.same_as(b);
}

IntSet Union(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::nothing();
  if (sets.size() == 1) return sets[0];
  Interval x = sets[0].cover_interval().as<IntervalSet>()->i;
  for (size_t i = 1; i < sets.size(); ++i) {
    IntSet s = sets[i].cover_interval();
    const Interval& y = s.as<IntervalSet>()->i;
    x.include(y);
  }
  x.max = ir::Simplify(x.max);
  x.min = ir::Simplify(x.min);
  return IntervalSet::make(x);
}

IntSet Intersect(const Array<IntSet>& sets) {
  Interval x = sets[0].cover_interval().as<IntervalSet>()->i;
  for (size_t i = 1; i < sets.size(); ++i) {
    Interval y = sets[i].cover_interval().as<IntervalSet>()->i;
    x = Interval::make_intersection(x, y);
  }
  return IntervalSet::make(x);
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

// interval related.
template<typename OP>
inline IntSet CombineInterval(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<OP>(a.min, b.min));
  }
  LOG(WARNING) << "Return Everything in CombineInterval " << OP::_type_key;
  return IntSet::everything();
}

template<>
inline IntSet CombineInterval<Add>(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<Add>(a.min, b.min));
  }
  Interval r = Interval::everything();
  if (a.has_lower_bound() && b.has_lower_bound()) {
    r.min = ComputeExpr<Add>(a.min, b.min);
  }
  if (a.has_upper_bound() && b.has_upper_bound()) {
    r.max = ComputeExpr<Add>(a.max, b.max);
  }
  return IntervalSet::make(r);
}

template<>
inline IntSet CombineInterval<Sub>(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<Sub>(a.min, b.min));
  }
  Interval r = Interval::everything();
  if (a.has_lower_bound() && b.has_upper_bound()) {
    r.min = ComputeExpr<Sub>(a.min, b.max);
  }
  if (a.has_upper_bound() && b.has_lower_bound()) {
    r.max = ComputeExpr<Sub>(a.max, b.min);
  }
  return IntervalSet::make(r);
}

template<>
inline IntSet CombineInterval<Mul>(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<Mul>(a.min, b.min));
  }
  if (a.is_single_point() && !b.is_single_point()) {
    std::swap(a, b);
  }
  if (b.is_single_point()) {
    if (is_zero(b.min)) return IntSet::single_point(0);
    if (is_one(b.min)) return IntervalSet::make(a);
    Expr e1 = a.has_lower_bound() ? ComputeExpr<Mul>(a.min, b.min) : a.min;
    Expr e2 = a.has_upper_bound() ? ComputeExpr<Mul>(a.max, b.min) : a.max;
    // no relaxation is needed in here due to set is inclusive
    // TODO(tqchen): consider convert to StrideSet.
    if (is_positive_const(b.min)) {
      return IntervalSet::make(e1, e2);
    } else if (is_negative_const(b.min)) {
      return IntervalSet::make(e2, e1);
    } else if (a.is_bounded()) {
      Expr cmp = b.min >= make_zero(b.min.type().element_of());
      return IntervalSet::make(select(cmp, e1, e2), select(cmp, e2, e1));
    }
  }
  LOG(WARNING) << "Return Everything in CombineInterval Mul";
  return IntSet::everything();
}

template<>
inline IntSet CombineInterval<Div>(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<Div>(a.min, b.min));
  }
  if (b.is_single_point()) {
    if (is_zero(b.min)) {
      LOG(FATAL) << "Divide by zero in CombineInterval Div";
    }
    if (is_one(b.min)) return IntervalSet::make(a);
    Expr e1 = a.has_lower_bound() ? ComputeExpr<Div>(a.min, b.min) : a.min;
    Expr e2 = a.has_upper_bound() ? ComputeExpr<Div>(a.max, b.min) : a.max;
    // no relaxation is needed in here due to set is inclusive
    if (is_positive_const(b.min)) {
      return IntervalSet::make(e1, e2);
    } else if (is_negative_const(b.min)) {
      return IntervalSet::make(e2, e1);
    } else if (a.is_bounded()) {
      Expr cmp = b.min >= make_zero(b.min.type().element_of());
      return IntervalSet::make(select(cmp, e1, e2), select(cmp, e2, e1));
    }
  }
  LOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntSet::everything();
}

template<>
inline IntSet CombineInterval<Mod>(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<Mod>(a.min, b.min));
  }
  if (b.is_single_point()) {
    Expr divisor = b.min;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    return IntervalSet::make(make_zero(divisor.type()), divisor - 1);
  }

  LOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntSet::everything();
}

template<>
inline IntSet CombineInterval<Max>(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<Max>(a.min, b.min));
  }
  return IntervalSet::make(Interval::make_max(a.min, b.min),
                           Interval::make_max(a.max, b.max));
}

template<>
inline IntSet CombineInterval<Min>(Interval a, Interval b) {
  if (a.is_single_point() && b.is_single_point()) {
    return IntSet::single_point(ComputeExpr<Min>(a.min, b.min));
  }
  return IntervalSet::make(Interval::make_min(a.min, b.min),
                           Interval::make_min(a.max, b.max));
}

template<typename OP>
inline IntSet CombineInterval_(IntSet a, IntSet b) {
  return CombineInterval<OP>(
      a.as<IntervalSet>()->i, b.as<IntervalSet>()->i);
}

// stride related
inline IntSet AsStrideSet(IntSet a) {
  if (a.as<StrideSet>()) return a;
  const IntervalSet* s = a.as<IntervalSet>();
  CHECK(s->i.is_bounded());
  NodePtr<StrideSet> n = make_node<StrideSet>();
  n->base = s->i;
  return IntSet(n);
}
template<typename OP>
inline IntSet CombineSets(IntSet a, IntSet b) {
  return CombineInterval_<OP>(a.cover_interval(), b.cover_interval());
}

template<>
inline IntSet CombineSets<Add>(IntSet a, IntSet b) {
  const IntervalSet* a_int = a.as<IntervalSet>();
  const IntervalSet* b_int = b.as<IntervalSet>();
  if (a_int && is_zero(a_int->i.min)) return b;
  if (b_int && is_zero(b_int->i.min)) return a;
  a = AsStrideSet(a);
  b = AsStrideSet(b);
  const StrideSet* a_stride = a.as<StrideSet>();
  const StrideSet* b_stride = b.as<StrideSet>();
  auto n = make_node<StrideSet>(*a_stride);
  for (size_t i = 0; i < b_stride->extents.size(); ++i) {
    n->extents.push_back(b_stride->extents[i]);
    n->strides.push_back(b_stride->strides[i]);
  }
  n->base = CombineInterval<Add>(
      a_stride->base, b_stride->base).as<IntervalSet>()->i;
  return IntSet(n);
}

inline IntSet NegateSet(IntSet a) {
  const IntervalSet* a_int = a.as<IntervalSet>();
  if (a_int) {
    if (a_int->i.is_single_point()) {
      return IntSet::single_point(-a_int->i.min);
    } else {
      Interval r = Interval::everything();
      if (a_int->i.has_upper_bound()) {
        r.min = -(a_int->i.max);
      }
      if (a_int->i.has_lower_bound()) {
        r.max = -(a_int->i.min);
      }
      return IntervalSet::make(r);
    }
  } else {
    return NegateSet(a.cover_interval());
  }
}

template<>
inline IntSet CombineSets<Sub>(IntSet a, IntSet b) {
  return CombineSets<Add>(a, NegateSet(b));
}

TVM_DECLARE_LOGICAL_OP(And);
TVM_DECLARE_LOGICAL_OP(Or);
TVM_DECLARE_LOGICAL_OP(EQ);
TVM_DECLARE_LOGICAL_OP(NE);
TVM_DECLARE_LOGICAL_OP(GE);
TVM_DECLARE_LOGICAL_OP(GT);
TVM_DECLARE_LOGICAL_OP(LE);
TVM_DECLARE_LOGICAL_OP(LT);
TVM_DECLARE_LOGICAL_OP(Not);

// generic combine operations of two sets
template<typename OP>
inline IntSet Combine(const IntSet& a, const IntSet &b) {
  if (is_logical_op<OP>::value) {
    return IntervalSet::make(0, 1);
  }
  const IntervalSet* a_int = a.as<IntervalSet>();
  const IntervalSet* b_int = b.as<IntervalSet>();
  if (a_int && a_int->i.is_everything()) return a;
  if (b_int && b_int->i.is_everything()) return b;
  if (a_int && b_int) {
    return CombineInterval<OP>(a_int->i, b_int->i);
  }
  if (a_int && !(a_int->i.is_bounded())) {
    return CombineInterval_<OP>(a, b.cover_interval());
  }
  if (b_int && !(b_int->i.is_bounded())) {
    return CombineInterval_<OP>(a.cover_interval(), b);
  }
  return CombineSets<OP>(a, b);
}

class IntSetEvaluator :
      public ExprFunctor<IntSet(const Expr&, const Expr&)> {
 public:
  explicit IntSetEvaluator(
      const std::unordered_map<const Variable*, IntSet>& dom_map,
      bool eval_vec = false)
      : dom_map_(dom_map), eval_vec_(eval_vec) {}
  // Evaluate.
  IntSet Eval(const Expr& e) {
    return this->VisitExpr(e, e);
  }
  IntSet VisitExpr_(const IntImm* op, const Expr& e) final {
    return IntSet::single_point(e);
  }
  IntSet VisitExpr_(const UIntImm* op, const Expr& e) final {
    return IntSet::single_point(e);
  }
  IntSet VisitExpr_(const Variable* op, const Expr& e) final {
    auto it = dom_map_.find(op);
    if (it != dom_map_.end()) {
      return it->second;
    } else {
      return IntSet::single_point(e);
    }
  }
  IntSet VisitExpr_(const Add* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Sub* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Mul* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Div* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Mod* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Min* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Max* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const EQ* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const NE* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const LT* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const LE* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const GT* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const GE* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const And* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Or* op, const Expr& e) final {
    return Binary(op, e);
  }
  IntSet VisitExpr_(const Ramp* op, const Expr& e) final {
    CHECK(eval_vec_);
    IntSet base = Eval(op->base);
    int vstride;
    if (GetConstInt(op->stride, &vstride)) {
      Type t = op->base.type();
      if (vstride > 0) {
        return Combine<Add>(
            base,
            IntSet::interval(make_zero(t),
                             make_const(t, vstride * op->lanes -1)));
      } else {
        return Combine<Add>(
            base,
            IntSet::interval(make_const(t, vstride * op->lanes + 1),
                             make_zero(t)));
      }
    }
    LOG(WARNING) << "cannot evaluate set on expression " << e;
    return IntSet::everything();
  }
  IntSet VisitExpr_(const Broadcast* op, const Expr& e) final {
    CHECK(eval_vec_);
    return Eval(op->value);
  }
  IntSet VisitExprDefault_(const Node* op, const Expr& e) final {
    LOG(WARNING) << "cannot evaluate set type " << e->type_key();
    return IntSet::everything();
  }

 private:
  template<typename T>
  inline IntSet Binary(const T* op, const Expr& e) {
    IntSet a = this->Eval(op->a);
    IntSet b = this->Eval(op->b);
    if (MatchPoint(a, op->a) && MatchPoint(b, op->b)) {
      return IntSet::single_point(e);
    }
    return Combine<T>(a, b);
  }

  const std::unordered_map<const Variable*, IntSet>& dom_map_;
  bool eval_vec_{false};
};

IntSet EvalSet(Expr e,
               const std::unordered_map<const Variable*, IntSet>& dom_map) {
  return IntSetEvaluator(dom_map, false).Eval(e);
}

IntSet IntSet::vector(Expr x) {
  std::unordered_map<const Variable*, IntSet> dmap;
  return IntSetEvaluator(dmap, true).Eval(x);
}

IntSet EvalSet(Expr e,
               const Map<IterVar, IntSet>& dom_map) {
  std::unordered_map<const Variable*, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap[kv.first->var.as<Variable>()] = kv.second;
  }
  return EvalSet(e, dmap);
}

IntSet EvalSet(Range r,
               const std::unordered_map<const Variable*, IntSet>& dom_map) {
  IntSetEvaluator m(dom_map);
  IntSet min_set = m.Eval(r->min);
  IntSet ext_set = m.Eval(r->extent).cover_interval();
  const Interval& ei = ext_set.as<IntervalSet>()->i;
  if (!ei.has_upper_bound()) return IntSet::everything();
  ext_set = IntervalSet::make(make_zero(ei.max.type()), ComputeExpr<Sub>(ei.max, 1));
  return Combine<Add>(min_set, ext_set);
}

IntSet EvalSet(IntSet s,
               const std::unordered_map<const Variable*, IntSet>& dom_map) {
  IntSetEvaluator m(dom_map);
  s = s.cover_interval();
  const IntervalSet* s_int = s.as<IntervalSet>();
  Expr vmax = s_int->i.has_upper_bound() ?
      m.Eval(s_int->i.max).cover_interval().max() : s_int->i.max;
  Expr vmin = s_int->i.has_lower_bound() ?
      m.Eval(s_int->i.min).cover_interval().min() : s_int->i.min;
  return IntervalSet::make(vmin, vmax);
}

class SubExprIntSetEvaluator : public IntSetEvaluator {
 public:
  explicit SubExprIntSetEvaluator(
      const std::unordered_map<const Variable*, IntSet>& dom_map)
      : IntSetEvaluator(dom_map) {}

  IntSet VisitExpr(const Expr& n, const Expr& e) final {
    IntSet ret = IntSetEvaluator::VisitExpr(n, e);
    expr_map[n] = ret;
    return ret;
  }

  ExprIntSetMap expr_map;
};

ExprIntSetMap EvalSetForEachSubExpr(Expr e,
    const std::unordered_map<const Variable*, IntSet>& dom_map) {
  SubExprIntSetEvaluator m(dom_map);
  m.Eval(e);
  return m.expr_map;
}

IntSet EvalSet(Range r,
               const Map<IterVar, IntSet>& dom_map) {
  std::unordered_map<const Variable*, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap[kv.first->var.as<Variable>()] = kv.second;
  }
  return EvalSet(r, dmap);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IntervalSet>([](const IntervalSet *op, IRPrinter *p) {
    p->stream << "interval-set"
              << "[" << op->i.min << ", "
              << op->i.max << ']';
  });

}  // namespace arith
}  // namespace tvm
