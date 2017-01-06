/*!
 *  Copyright (c) 2016 by Contributors
 * \file int_set.cc
 * \brief The integer set functions
 */
#include <tvm/ir.h>
#include "./int_set.h"

namespace tvm {
namespace schedule {

using namespace ir;

/*!
 * \brief Internal node container of int set.
 */
class IntSetNode : public Node {
 public:
  /*! \brief The base range scope */
  Range base;
  /*! \brief additional strided domain */
  Array<Range> domain;
  /*! \brief The stride of each strided domain */
  Array<Expr> stride;
  /*!
   * \brief The concrete set,
   *  used when concrete execution is enabled.
   */
  std::vector<int32_t> concrete;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("base", &base);
    v->Visit("domain", &domain);
    v->Visit("stride", &stride);
  }

  static constexpr const char* _type_key = "IntSet";
  TVM_DECLARE_NODE_TYPE_INFO(IntSetNode);
};

TVM_REGISTER_NODE_TYPE(IntSetNode);

namespace {

inline bool Match(const Expr& e, int64_t value) {
  const ir::IntImm* v = e.as<ir::IntImm>();
  return v != nullptr && v->value;
}

// whether a exactly matches b.
inline bool Match(const IntSet& a,
                  const Range& b) {
  if (a->base == b &&
      a->domain.size() == 0 &&
      a->concrete.size() == 0) {
    return true;
  } else {
    return false;
  }
}

// whether a exactly matches b.
inline bool Match(const IntSet& a,
                  const Expr& b) {
  if (a->domain.size() == 0 &&
      a->concrete.size() == 0) {
    return Match(a->base->extent, 1) && a->base->min.same_as(b);
  } else {
    return false;
  }
}

inline bool IsNumber(const IntSet& s) {
  if (s->domain.size() != 0) return false;
  if (s->concrete.size() != 0) {
    return s->concrete.size() == 1;
  }
  return Match(s->base->extent, 1);
}

inline Expr AsNumber(const IntSet& s) {
  return s->base->min;
}

// set combination rule by operators
template<typename T>
inline IntSet BinaryCombine(IntSet a, IntSet b) {
  LOG(WARNING) << "cannot evaluate binary op " << T::_type_key;
  return IntSet::make_all_set();
}

template<>
inline IntSet BinaryCombine<Add>(IntSet a, IntSet b) {
  auto n = std::make_shared<IntSetNode>(*(a.operator->()));
  for (size_t i = 0; i < b->domain.size(); ++i) {
    n->domain.push_back(b->domain[i]);
    n->stride.push_back(b->stride[i]);
  }

  if (IsNumber(a)) {
    n->base = Range::make_with_min_extent(
        a->base->min + b->base->min,
        b->base->extent);
  } else if (IsNumber(b)) {
    n->base = Range::make_with_min_extent(
        a->base->min + b->base->min,
        a->base->extent);
  } else {
    n->base = Range::make_with_min_extent(
        a->base->min + b->base->min,
        a->base->extent + b->base->extent - 1);
  }
  return IntSet(n);
}

inline Range Negation(Range a) {
  if (Match(a->extent, 1)) {
    return Range::make_with_min_extent(-a->min, a->extent);
  } else {
    return Range::make_with_min_extent(-(a->min + a->extent - 1), a->extent);
  }
}

inline IntSet Negation(IntSet a) {
  CHECK_EQ(a->concrete.size(), 0U);
  auto n = std::make_shared<IntSetNode>();
  n->base = Negation(a->base);
  for (size_t i = 0; i < a->domain.size(); ++i) {
    n->domain.push_back(Negation(a->domain[i]));
    n->stride.push_back(a->stride[i]);
  }
  return IntSet(a);
}

template<>
inline IntSet BinaryCombine<Sub>(IntSet a, IntSet b) {
  return BinaryCombine<Add>(a, Negation(b));
}

inline IntSet BinaryMul(IntSet a, Expr b) {
  // copy construct
  if (Match(b, 1)) return a;
  if (Match(b, -1)) return Negation(a);
  auto n = std::make_shared<IntSetNode>();
  n->base = Range::make_with_min_extent(0, 1);
  n->domain.push_back(a->base);
  n->stride.push_back(b);
  for (size_t i = 0; i < a->domain.size(); ++i) {
    n->domain.push_back(a->domain[i]);
    n->stride.push_back(a->stride[i] * b);
  }
  return IntSet(a);
}

template<>
inline IntSet BinaryCombine<Mul>(IntSet a, IntSet b) {
  if (IsNumber(a)) {
    return BinaryMul(a, AsNumber(b));
  } else if (IsNumber(b)) {
    return BinaryMul(b, AsNumber(a));
  } else {
    return IntSet::make_all_set();
  }
}

}  // namespace

inline const IntSetNode* IntSet::operator->() const {
  return static_cast<const IntSetNode*>(node_.get());
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<IntSetNode>([](const IntSetNode *op, IRPrinter *p) {
    p->stream << "int-set(base=";
    p->print(op->base);
    p->stream << ')';
  });

IntSet IntSet::make_range(Range dom) {
  auto n = std::make_shared<IntSetNode>();
  n->base = dom;
  return IntSet(n);
}

Range IntSet::GetCoverRange() const {
  const IntSetNode* s = operator->();
  CHECK(s != nullptr) << "empty set";
  if (s->domain.size() == 0 && s->concrete.size() == 0) {
    return s->base;
  }
  LOG(FATAL) << "not yet implemented";
  return Range();
}

IntSet IntSet::make_point(Expr point) {
  return IntSet::make_range(Range::make_with_min_extent(point, 1));
}

IntSet IntSet::make_all_set() {
  LOG(FATAL) << "TODO";
  return IntSet();
}

IntSet Union(const Array<IntSet>& set) {
  if (set.size() == 1) return set[0];
  LOG(FATAL) << "TODO";
  return IntSet();
}

void PassUp(const SplitNode* s,
            const std::unordered_map<IterVar, Range>& dom_map,
            const IntSet& outer,
            const IntSet& inner,
            IntSet* parent) {
  if (dom_map.count(s->outer) &&
      dom_map.count(s->inner) &&
      dom_map.count(s->parent) &&
      Match(outer, dom_map.at(s->outer)) &&
      Match(inner, dom_map.at(s->inner))) {
    *parent = IntSet::make_range(dom_map.at(s->parent));
    return;
  }
  // copy construct
  auto n = std::make_shared<IntSetNode>(*(inner.operator->()));

  if (IsNumber(outer)) {
    // shift the base offset
    n->base = Range::make_with_min_extent(
        AsNumber(outer) * s->factor + inner->base->min,
        inner->base->extent);
    *parent = IntSet(n);
  } else {
    // default use all domains in the data.
    n->domain.push_back(outer->base);
    n->stride.push_back(s->factor);
    for (size_t i = 0; i < outer->domain.size(); ++i) {
      n->domain.push_back(outer->domain[i]);
      n->stride.push_back(outer->stride[i] * s->factor);
    }
  }
}

void PassUp(const FuseNode* s,
            const std::unordered_map<IterVar, Range>& dom_map,
            const IntSet& fused,
            IntSet* outer,
            IntSet* inner) {
  CHECK(dom_map.count(s->outer));
  CHECK(dom_map.count(s->inner));
  CHECK(dom_map.count(s->fused));

  if (Match(fused, dom_map.at(s->fused))) {
    *outer = IntSet::make_range(dom_map.at(s->outer));
    *inner = IntSet::make_range(dom_map.at(s->inner));
    return;
  }

  if (IsNumber(fused)) {
    Expr value = AsNumber(fused);
    Expr factor = dom_map.at(s->outer)->extent;
    *outer = IntSet::make_point(value / factor);
    *inner = IntSet::make_point(value % factor);
  } else {
    LOG(WARNING) << "use fallback inference rule in fuse";
    // simply use the entire set, this rule can be enhanced.
    *outer = IntSet::make_range(dom_map.at(s->outer));
    *inner = IntSet::make_range(dom_map.at(s->inner));
    return;
  }
}

namespace {
// evaluator to evaluate the int set
class IRSetEvaluator {
 public:
  inline IntSet Eval(Expr expr) {
    static const FType& f = vtable();
    if (f.can_dispatch(expr)) {
      return f(expr, expr, this);
    } else {
      LOG(WARNING) << "cannot evaluate set type " << expr->type_key();
      return IntSet::make_all_set();
    }
  }

  using FType = tvm::IRFunctor<IntSet (const NodeRef&, const Expr&, IRSetEvaluator *)>;
  static FType& vtable() {  // NOLINT(*)
    static FType inst; return inst;
  }

  std::unordered_map<const Variable*, IntSet> dom_map;
};

inline IntSet ConstOp(const NodeRef&, const Expr& e, IRSetEvaluator*) {
  return IntSet::make_point(e);
}

TVM_STATIC_IR_FUNCTOR(IRSetEvaluator, vtable)
.set_dispatch<IntImm>(ConstOp)
.set_dispatch<UIntImm>(ConstOp)
.set_dispatch<FloatImm>(ConstOp);

TVM_STATIC_IR_FUNCTOR(IRSetEvaluator, vtable)
.set_dispatch<Variable>([](const Variable* op, const Expr& e, IRSetEvaluator* m) {
    auto it = m->dom_map.find(op);
    if (it != m->dom_map.end()) {
      return it->second;
    } else {
      return IntSet::make_point(e);
    }
  });

// binary operator
template<typename T>
inline IntSet Binary(const T* op, const Expr& e, IRSetEvaluator* m) {
  IntSet a = m->Eval(op->a);
  IntSet b = m->Eval(op->b);
  if (IsNumber(a) && IsNumber(b)) {
    if (Match(a, op->a) &&
        Match(b, op->b)) {
      return IntSet::make_point(e);
    } else {
      return IntSet::make_point(T::make(AsNumber(a), AsNumber(b)));
    }
  } else {
    return BinaryCombine<T>(a, b);
  }
}

TVM_STATIC_IR_FUNCTOR(IRSetEvaluator, vtable)
.set_dispatch<Add>(Binary<Add>)
.set_dispatch<Sub>(Binary<Sub>)
.set_dispatch<Mul>(Binary<Mul>)
.set_dispatch<Div>(Binary<Div>)
.set_dispatch<Mod>(Binary<Mod>)
.set_dispatch<Min>(Binary<Min>)
.set_dispatch<Max>(Binary<Max>);

// use simply bound for logical expressions for now.
inline IntSet Logical(const NodeRef&, const Expr& e, IRSetEvaluator*) {
  return IntSet::make_range(Range::make_with_min_extent(0, 2));
}

TVM_STATIC_IR_FUNCTOR(IRSetEvaluator, vtable)
.set_dispatch<EQ>(Logical)
.set_dispatch<NE>(Logical)
.set_dispatch<LT>(Logical)
.set_dispatch<LE>(Logical)
.set_dispatch<GT>(Logical)
.set_dispatch<GE>(Logical)
.set_dispatch<And>(Logical)
.set_dispatch<Or>(Logical);

}  // namespace

IntSet EvalSet(Expr e,
               const Map<IterVar, IntSet>& dom_map) {
  IRSetEvaluator m;
  for (auto kv : dom_map) {
    m.dom_map[kv.first->var.as<Variable>()] = kv.second;
  }
  return m.Eval(e);
}

}  // namespace schedule
}  // namespace tvm
