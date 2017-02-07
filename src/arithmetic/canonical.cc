/*!
 *  Copyright (c) 2017 by Contributors
 * \file canonical.cc
 * \brief Canonicalize simplification.
 */
#include <tvm/ir_mutator.h>
#include "./int_set.h"
#include "./canonical.h"
#include "./compute_expr.h"

namespace tvm {
namespace arith {
using namespace ir;

// Canonical entry for communicative ops.
struct ComExprEntry {
  // the value of the expression.
  Expr value;
  // the level of the expression.
  int level{0};
  // The integer scale on value
  int64_t scale{1};

  ComExprEntry() {}
  ComExprEntry(Expr value, int level)
      : value(value), level(level) {}
  inline bool operator<(const ComExprEntry& other) const {
    if (level < other.level) return true;
    if (level > other.level) return false;
    return value.get() < other.value.get();
  }
};

// canonical expression for communicative expression.
struct ComExprNode {
  // base constant value.
  int64_t base{0};
  // The values to be sumed.
  std::vector<ComExprEntry> elem;
};

// canonical communicative expression
struct ComExpr {
 public:
  // constructor
  ComExpr() {}
  explicit ComExpr(std::shared_ptr<ComExprNode> ptr) : ptr_(ptr) {}
  // get member
  ComExprNode* operator->() const {
    return ptr_.get();
  }
  void reset() {
    ptr_.reset();
  }
  bool defined() const {
    return ptr_.get() != nullptr;
  }
  // comparator
  bool operator<(const ComExpr& b) const {
    const ComExpr& a = *this;
    if (a->base < b->base) return true;
    if (a->base > b->base) return false;
    if (a->elem.size() < b->elem.size()) return true;
    if (a->elem.size() > b->elem.size()) return false;
    for (size_t i = 0; i < a->elem.size(); ++i) {
      const ComExprEntry& ea = a->elem[i];
      const ComExprEntry& eb = b->elem[i];
      if (ea.level < eb.level) return true;
      if (ea.level > eb.level) return false;
      if (ea.value.get() < eb.value.get()) return true;
      if (ea.value.get() > eb.value.get()) return false;
      if (ea.scale < eb.scale) return true;
      if (ea.scale > eb.scale) return false;
    }
    return false;
  }
  // equality
  bool operator==(const ComExpr& b) const {
    const ComExpr& a = *this;
    if (a->base != b->base) return false;
    if (a->elem.size() != b->elem.size()) return false;
    for (size_t i = 0; i < a->elem.size(); ++i) {
      const ComExprEntry& ea = a->elem[i];
      const ComExprEntry& eb = b->elem[i];
      if (ea.level != eb.level) return false;
      if (ea.value.get() != eb.value.get()) return false;
      if (ea.scale != eb.scale) return false;
    }
    return true;
  }

 private:
  std::shared_ptr<ComExprNode> ptr_;
};

template<typename T>
inline Expr Binary_(const T* op,
                    const Expr& e,
                    Expr a, Expr b) {
  if (a.same_as(op->a) && b.same_as(op->b)) {
    return e;
  } else {
    return T::make(a, b);
  }
}

template<typename T>
inline Expr Binary(
    const T* op, const Expr& e, IRMutator* m) {
  return Binary_(op, e, m->Mutate(op->a), m->Mutate(op->b));
}

// internal of canonical engine.
class Canonical::Internal : public IRMutator {
 public:
  // stack entry.
  struct StackEntry {
    int max_level{0};
    bool has_side_effect{false};
  };
  // aggressively canonicalized expression
  struct CacheEntry {
    // The canonical value of the expression.
    Expr value;
    // The level of the expression.
    int max_level{0};
    // whether the expression might have side effect.
    bool has_side_effect{false};
    // if not null, corresponds to to sum
    ComExpr sum;
    // reset the return entry.
    void reset() {
      sum.reset();
    }
    // as sum expr
    ComExpr AsSum() const {
      if (sum.defined()) return sum;
      const int64_t *v1 = as_const_int(value);
      const uint64_t *v2 = as_const_uint(value);
      std::shared_ptr<ComExprNode> n = std::make_shared<ComExprNode>();
      if (v1) {
        n->base = *v1;
      } else if (v2) {
        CHECK_LE(*v2,
               static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
        n->base = static_cast<int64_t>(*v2);
      } else {
        n->elem.push_back(ComExprEntry(value, max_level));
      }
      return ComExpr(n);
    }
  };
  // Set range and level of var.
  void SetRange(Var v, Range r, int level) {
    var_range_[v.get()] = IntSet::range(r);
    var_level_[v.get()] = level;
    var_rec_.push_back(v);
  }
  // functions
  Stmt Mutate(Stmt stmt) final {
    stmt = IRMutator::Mutate(stmt);
    return stmt;
  }
  Expr MutateExpr_(Expr expr) {
    static const FMutateExpr& f = Internal::vtable_expr();
    stack_.push_back(StackEntry());
    expr =  (f.can_dispatch(expr) ?
            f(expr, expr, this) : IRMutator::Mutate(expr));
    // update result of parent automatically during pop
    if (stack_.size() > 1) {
      StackEntry& back = stack_[stack_.size() - 1];
      StackEntry& prev = stack_[stack_.size() - 2];
      prev.max_level = std::max(prev.max_level, back.max_level);
      if (back.has_side_effect) prev.has_side_effect = true;
    }
    // copy result from stack
    ret_entry_.has_side_effect = stack_.back().has_side_effect;
    ret_entry_.max_level = stack_.back().max_level;
    stack_.pop_back();
    CHECK(expr.defined());
    return expr;
  }
  // call produce to get a cache entry.
  CacheEntry Produce(Expr expr) {
    ret_entry_.reset();
    ret_entry_.value = MutateExpr_(expr);
    CacheEntry ret  = ret_entry_;
    ret_entry_.reset();
    return ret;
  }
  Expr Mutate(Expr expr) final {
    ret_entry_.reset();
    expr = MutateExpr_(expr);
    ret_entry_.reset();
    return expr;
  }

  // Check whether do special canonicalization.
  bool EnableOpt(Type t) const {
    return (t.lanes() == 1 && (t.is_int() || t.is_uint()));
  }
  // Add
  Expr Mutate_(const Add* op, const Expr& e) {
    if (!EnableOpt(op->type)) {
      return Binary(op, e, this);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    return SumAdd(a, b, +1);
  }
  // Sub
  Expr Mutate_(const Sub* op, const Expr& e) {
    if (!EnableOpt(op->type)) {
      return Binary(op, e, this);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    return SumAdd(a, b, -1);
  }
  // Mul
  Expr Mutate_(const Mul* op, const Expr& e) {
    if (!EnableOpt(op->type)) {
      return Binary(op, e, this);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    if (is_const(a.value) && is_const(b.value)) {
      return ComputeExpr<Mul>(a.value, b.value);
    } else if (is_const(a.value)) {
      return SumMulConst(b.AsSum(), a.value);
    } else if (is_const(b.value)) {
      return SumMulConst(a.AsSum(), b.value);
    } else {
      return Binary_(op, e, a.value, b.value);
    }
  }
  // Variable
  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_level_.find(op);
    if (it != var_level_.end()) {
      stack_.back().max_level = it->second;
    }
    return IRMutator::Mutate_(op, e);
  }
  // comparison
  Expr Mutate_(const LT* op, const Expr& e) {
    if (!EnableOpt(op->a.type())) {
      return Binary(op, e, this);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    Expr b_sub_a = SumAdd(b, a, -1);
    if (EvalSet(b_sub_a, var_range_).can_prove_positive()) {
      return make_const(op->type, true);
    } else {
      return Binary_(op, e, a.value, b.value);
    }
  }
  // Call
  Expr Mutate_(const Call* op, const Expr& e) final {
    if (!op->is_pure()) {
      stack_.back().has_side_effect = true;
    }
    return IRMutator::Mutate_(op, e);
  }
  // For
  Stmt Mutate_(const For* op, const Stmt& s) {
    ++level_counter_;
    Var loop_var(op->loop_var.node_);
    this->SetRange(loop_var,
                   Range::make_with_min_extent(op->min, op->extent),
                   level_counter_);
    Stmt stmt = IRMutator::Mutate_(op, s);
    --level_counter_;
    return stmt;
  }
  // AttrStmt
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    if (op->type_key == "thread_extent") {
      ++level_counter_;
      IterVar iv(op->node.node_);
      CHECK_NE(iv->thread_tag.length(), 0U);
      if (!var_level_.count(iv->var.get())) {
        this->SetRange(iv->var,
                       Range::make_with_min_extent(0, op->value),
                       level_counter_);
      }
      Stmt stmt = IRMutator::Mutate_(op, s);
      --level_counter_;
      return stmt;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  // The simplify statement.
  static FMutateExpr& vtable_expr() {  // NOLINT(*)
    static FMutateExpr inst; return inst;
  }

 private:
  // return entry
  CacheEntry ret_entry_;
  // internal information stack
  std::vector<StackEntry> stack_;
  // cache sum
  std::map<ComExpr, CacheEntry> cache_sum_;
  // range of each var
  std::unordered_map<const Variable*, IntSet> var_range_;
  // level of each var
  std::unordered_map<const Variable*, int> var_level_;
  // record history vars, to avoid false positive.
  std::vector<Var> var_rec_;
  // level counter
  int level_counter_{0};
  // subroutine to do produce
  Expr SumMulConst(ComExpr a, Expr v) {
    int64_t value = 0;
    const int64_t *v1 = as_const_int(v);
    const uint64_t *v2 = as_const_uint(v);
    CHECK(v1 || v2);
    if (v1) {
      value = *v1;
    } else if (v2) {
      CHECK_LE(*v2,
               static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
      value = static_cast<int64_t>(*v2);
    }

    if (value == 0) {
      return make_zero(v.type());
    }
    std::shared_ptr<ComExprNode> vsum =
        std::make_shared<ComExprNode>(*a.operator->());
    vsum->base *= value;
    for (auto& e : vsum->elem) {
      e.scale *= value;
    }
    ret_entry_.max_level = stack_.back().max_level;
    ret_entry_.has_side_effect = stack_.back().has_side_effect;
    ret_entry_.sum = ComExpr(vsum);
    auto it = cache_sum_.find(ret_entry_.sum);
    if (it != cache_sum_.end()) {
      ret_entry_ = it->second;
    } else {
      ret_entry_.value = Sum2Expr(ret_entry_.sum, v.type());
      cache_sum_[ret_entry_.sum] = ret_entry_;
    }
    return ret_entry_.value;
  }
  // add two ComExpr together
  ComExpr SumAdd_(const ComExpr& suma,
                  const ComExpr& sumb,
                  int bscale) {
    std::shared_ptr<ComExprNode> n = std::make_shared<ComExprNode>();
    n->base = suma->base + sumb->base;
    // merge of suma and sumb;
    size_t i = 0, j = 0;
    while (i < suma->elem.size() && j < sumb->elem.size()) {
      const auto& a = suma->elem[i];
      const auto& b = sumb->elem[j];
      if (a.value.same_as(b.value)) {
        CHECK_EQ(a.level, b.level);
        ComExprEntry e = a;
        e.scale = a.scale + b.scale * bscale;
        if (e.scale != 0) {
          n->elem.push_back(e);
        }
        ++i; ++j;
      } else if (a < b) {
        n->elem.push_back(a);
        ++i;
      } else {
        ComExprEntry e = b;
        e.scale *= bscale;
        n->elem.push_back(e);
        ++j;
      }
    }
    for (; i < suma->elem.size(); ++i) {
      n->elem.push_back(suma->elem[i]);
    }
    for (; j < sumb->elem.size(); ++j) {
      ComExprEntry e = sumb->elem[j];
      e.scale *= bscale;
      n->elem.push_back(e);
    }
    return ComExpr(n);
  }
  // subroutine to do produce
  Expr SumAdd(CacheEntry a, CacheEntry b, int bscale) {
    ret_entry_.sum = SumAdd_(a.AsSum(), b.AsSum(), bscale);
    CHECK_NE(stack_.size(), 0U);
    ret_entry_.max_level = stack_.back().max_level;
    ret_entry_.has_side_effect = stack_.back().has_side_effect;
    auto it = cache_sum_.find(ret_entry_.sum);
    if (it != cache_sum_.end()) {
      ret_entry_ = it->second;
    } else {
      ret_entry_.value = Sum2Expr(ret_entry_.sum, a.value.type());
      cache_sum_[ret_entry_.sum] = ret_entry_;
    }
    return ret_entry_.value;
  }
  // convert sum to expr
  Expr Sum2Expr(const ComExpr& com, Type t) {
    Expr vsum;
    if (com->base != 0) {
      vsum = make_const(t, com->base);
    }
    for (const ComExprEntry& e : com->elem) {
      if (e.scale > 0) {
        Expr v = e.value;
        if (e.scale != 1) {
          v = Mul::make(v, make_const(t, e.scale));
        }
        if (vsum.defined()) {
          vsum = Add::make(vsum, v);
        } else {
          vsum = v;
        }
      }
    }
    for (const ComExprEntry& e : com->elem) {
      if (e.scale < 0) {
        Expr v = e.value;
        if (e.scale != -1) {
          v = Mul::make(v, make_const(t, -e.scale));
        }
        if (vsum.defined()) {
          vsum = Sub::make(vsum, v);
        } else {
          vsum = Sub::make(make_zero(t), v);
        }
      }
    }
    if (vsum.defined()) {
      return vsum;
    } else {
      return make_zero(t);
    }
  }
};

using CInternal = Canonical::Internal;

#define DISPATCH_EXPR(OP)                                          \
  set_dispatch<OP>([](const OP *op, const Expr& e, IRMutator* p) { \
    return static_cast<CInternal*>(p)->Mutate_(op, e); })

TVM_STATIC_IR_FUNCTOR(CInternal, vtable_expr)
.DISPATCH_EXPR(Add)
.DISPATCH_EXPR(Sub)
.DISPATCH_EXPR(Mul)
.DISPATCH_EXPR(LT);


Canonical::Canonical()
    : ptr_(std::make_shared<Internal>()) {}

Expr Canonical::Simplify(Expr expr) {
  return ptr_->Mutate(expr);
}

Stmt Canonical::Simplify(Stmt stmt) {
  return ptr_->Mutate(stmt);
}

void Canonical::SetRange(Var v, Range r, int level) {
  ptr_->SetRange(v, r, level);
}
}  // namespace arith

namespace ir {
Stmt CanonicalSimplify(Stmt stmt) {
  return arith::Canonical().Simplify(stmt);
}

}  // namespace ir
}  // namespace tvm
