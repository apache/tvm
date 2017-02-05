/*!
 *  Copyright (c) 2017 by Contributors
 * \file int_set.cc
 * \brief The integer set functions
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <pass/Interval.h>
#include "./int_set.h"
#include "./compute_expr.h"

namespace tvm {
namespace arith {

using Halide::Internal::Interval;

using namespace ir;

/*! \brief Set of continuous interval */
struct IntervalSet : public IntSetNode {
  /*! \brief the internal interval*/
  Interval i;

  static IntSet make(Interval i) {
    std::shared_ptr<IntervalSet> n =
        std::make_shared<IntervalSet>();
    n->i = i;
    return IntSet(n);
  }
  static IntSet make(Expr min, Expr max) {
    std::shared_ptr<IntervalSet> n =
        std::make_shared<IntervalSet>();
    n->i.min = min;
    n->i.max = max;
    return IntSet(n);
  }

  static constexpr const char* _type_key = "IntervalSet";
  TVM_DECLARE_NODE_TYPE_INFO(IntervalSet);
};

/*!
 * \brief set represented by strided integers
 *  Reserved for cases where strided access is supported.
 */
struct StrideSet : public IntSetNode {
  /*! \brief the base inetrval */
  Interval base;
  /*! \brief additional extents in positive number */
  Array<Expr> extents;
  /*! \brief additional strides in positive number */
  Array<Expr> strides;

  static constexpr const char* _type_key = "StrideSet";
  TVM_DECLARE_NODE_TYPE_INFO(StrideSet);
};

inline IntSet IntSet::cover_interval() const {
  if ((*this).as<IntervalSet>()) return *this;
  const StrideSet* s =  (*this).as<StrideSet>();
  if (s) {
    CHECK_NE(s->extents.size(), 0U);
    Expr max = s->base.max;
    for (size_t i = 0; i < s->extents.size(); ++i) {
      max = max + s->extents[i] * s->strides[i] - s->strides[i];
    }
    return IntervalSet::make(s->base.min, max);
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
    return Range::make_with_min_extent(
        s_int->i.min, Simplify(s_int->i.max + 1 - s_int->i.min));
  }
  return max_range;
}

bool IntSet::is_everything() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  return (s_int && s_int->i.is_everything());
}

bool IntSet::is_single_point() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  return (s_int && s_int->i.is_single_point());
}

Expr IntSet::point_value() const {
  const IntervalSet* s_int = (*this).as<IntervalSet>();
  CHECK(s_int && s_int->i.is_single_point());
  return s_int->i.min;
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

// Check if a is created from b.
bool IntSet::match_range(const Range& b) const {
  const IntSet& a = *this;
  const IntervalSet* a_int = a.as<IntervalSet>();
  if (!a_int) return false;
  const Interval& i = a_int->i;
  if (!i.min.same_as(b)) return false;
  if (is_one(b->extent)) return i.is_single_point();
  if (is_positive_const(b->extent) && is_const(b->min)) {
    // deep equality
    return Equal(
        ComputeExpr<Sub>(ComputeExpr<Add>(b->extent, b->min), 1),
        a_int->i.max);
  }
  const Sub* sub = i.max.as<Sub>();
  if (!sub) return false;
  if (is_one(sub->b)) return false;
  const Add* add = sub->a.as<Add>();
  return add &&
      add->a.same_as(b->min) &&
      add->b.same_as(b->extent);
}

inline bool MatchPoint(const IntSet& a,
                       const Expr& b) {
  const IntervalSet* a_int = a.as<IntervalSet>();
  if (!a_int) return false;
  const Interval& i = a_int->i;
  return i.is_single_point() && i.min.same_as(b);
}

IntSet Union(const Array<IntSet>& set) {
  if (set.size() == 1) return set[0];
  Interval x = set[0].cover_interval().as<IntervalSet>()->i;
  for (size_t i = 1; i < set.size(); ++i) {
    x.include(set[i].cover_interval().as<IntervalSet>()->i);
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
    // This is relaxiation
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
  std::shared_ptr<StrideSet> n = std::make_shared<StrideSet>();
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
  auto n = std::make_shared<StrideSet>(*a_stride);
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

// Evaluator to evalute the epxression.
class IntSetEvaluator {
 public:
  inline IntSet Eval(Expr expr) {
    static const FType& f = vtable();
    if (f.can_dispatch(expr)) {
      return f(expr, expr, this);
    } else {
      LOG(WARNING) << "cannot evaluate set type " << expr->type_key();
      return IntSet::everything();
    }
  }

  using FType = tvm::IRFunctor<IntSet (const NodeRef&, const Expr&, IntSetEvaluator *)>;
  static FType& vtable() {  // NOLINT(*)
    static FType inst; return inst;
  }

  std::unordered_map<const Variable*, IntSet> dom_map;
};

inline IntSet ConstOp(const NodeRef&, const Expr& e, IntSetEvaluator*) {
  return IntSet::single_point(e);
}

TVM_STATIC_IR_FUNCTOR(IntSetEvaluator, vtable)
.set_dispatch<IntImm>(ConstOp)
.set_dispatch<UIntImm>(ConstOp)
.set_dispatch<FloatImm>(ConstOp);

TVM_STATIC_IR_FUNCTOR(IntSetEvaluator, vtable)
.set_dispatch<Variable>([](const Variable* op, const Expr& e, IntSetEvaluator* m) {
    auto it = m->dom_map.find(op);
    if (it != m->dom_map.end()) {
      return it->second;
    } else {
      return IntSet::single_point(e);
    }
  });

// binary operator
template<typename T>
inline IntSet Binary(const T* op, const Expr& e, IntSetEvaluator* m) {
  IntSet a = m->Eval(op->a);
  IntSet b = m->Eval(op->b);
  if (MatchPoint(a, op->a) && MatchPoint(b, op->b)) {
    return IntSet::single_point(e);
  }
  IntSet r = Combine<T>(a, b);
  return r;
}

TVM_STATIC_IR_FUNCTOR(IntSetEvaluator, vtable)
.set_dispatch<Add>(Binary<Add>)
.set_dispatch<Sub>(Binary<Sub>)
.set_dispatch<Mul>(Binary<Mul>)
.set_dispatch<Div>(Binary<Div>)
.set_dispatch<Mod>(Binary<Mod>)
.set_dispatch<Min>(Binary<Min>)
.set_dispatch<Max>(Binary<Max>)
.set_dispatch<EQ>(Binary<EQ>)
.set_dispatch<NE>(Binary<NE>)
.set_dispatch<LT>(Binary<LT>)
.set_dispatch<LE>(Binary<LE>)
.set_dispatch<GT>(Binary<GT>)
.set_dispatch<GE>(Binary<GE>)
.set_dispatch<And>(Binary<And>)
.set_dispatch<Or>(Binary<Or>);

IntSet EvalSet(Expr e,
               const Map<IterVar, IntSet>& dom_map) {
  IntSetEvaluator m;
  for (auto kv : dom_map) {
    m.dom_map[kv.first->var.as<Variable>()] = kv.second;
  }
  return m.Eval(e);
}

IntSet EvalSet(Range r,
               const Map<IterVar, IntSet>& dom_map) {
  IntSetEvaluator m;
  for (auto kv : dom_map) {
    m.dom_map[kv.first->var.as<Variable>()] = kv.second;
  }
  IntSet min_set = m.Eval(r->min);
  IntSet ext_set = m.Eval(r->extent).cover_interval();
  const Interval& ei = ext_set.as<IntervalSet>()->i;
  if (!ei.has_upper_bound()) return IntSet::everything();
  ext_set = IntervalSet::make(0, ComputeExpr<Sub>(ei.max, 1));
  return Combine<Add>(min_set, ext_set);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IntervalSet>([](const IntervalSet *op, IRPrinter *p) {
    p->stream << "interval-set["
              << "[" << op->i.min << ", "
              << op->i.max << ']';
  });


}  // namespace arith
}  // namespace tvm
