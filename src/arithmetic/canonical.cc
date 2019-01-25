/*!
 *  Copyright (c) 2017 by Contributors
 * \file canonical.cc
 * \brief Canonicalize simplification.
 */
#include <tvm/ir_mutator.h>
#include <tvm/arithmetic.h>
#include <tvm/ir_pass.h>
#include <algorithm>
#include <map>
#include <limits>
#include <vector>
#include "canonical.h"
#include "compute_expr.h"
#include "arithmetic/Simplify.h"

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
    // compare top operator of entries and sort on that if possible (fast check)
    if (value.type_index() < other.value.type_index()) return true;
    if (value.type_index() > other.value.type_index()) return false;
    // if none of the above distinguishes the terms, compare the expression tree of the entries.
    // This is a slower check.
    int compare_result = Compare(value, other.value);
    if (compare_result < 0) return true;
    if (compare_result > 0) return false;
    // it's a problem if we see identical entries at this point. They should've been merged earlier.
    LOG(WARNING) << "we should not have identical entries at this point";
    return false;
  }
};

// canonical expression for communicative expression.
struct ComExprNode : public NodeBase {
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
  explicit ComExpr(NodePtr<ComExprNode> ptr) : ptr_(ptr) {}
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
  NodePtr<ComExprNode> ptr_;
};

// binary comparison op.
struct BinaryExpr {
  int kind;
  Expr lhs, rhs;
  // comparator
  bool operator<(const BinaryExpr& b) const {
    if (kind < b.kind) return true;
    if (kind > b.kind) return false;
    if (lhs.get() < b.lhs.get()) return true;
    if (lhs.get() > b.lhs.get()) return false;
    return rhs.get() < b.rhs.get();
  }
  // equality
  bool operator==(const BinaryExpr& b) const {
    return kind == b.kind &&
        lhs.same_as(b.lhs) &&
        rhs.same_as(b.rhs);
  }
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

// internal of canonical engine.
class Canonical::Internal : public IRMutator {
 public:
  explicit Internal(Map<Var, Range> vrange) {
    for (auto kv : vrange) {
      SetRange(kv.first, kv.second, 0);
    }
  }
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
      auto n = make_node<ComExprNode>();
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
    stack_.push_back(StackEntry());
    expr = IRMutator::Mutate(expr);
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
    if (const IntImm* op = expr.as<IntImm>()) {
      return Mutate_(op, expr);
    }
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
  // Max
  Expr Mutate_(const Max* op, const Expr& e) final {
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    return Binary(op, e);
  }
  // Min
  Expr Mutate_(const Min* op, const Expr& e) final {
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    return Binary(op, e);
  }
  // Add
  Expr Mutate_(const Add* op, const Expr& e) final {
    if (!EnableOpt(op->type)) {
      return Binary(op, e);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    return SumAdd(a, b, +1);
  }
  // Sub
  Expr Mutate_(const Sub* op, const Expr& e) final {
    if (!EnableOpt(op->type)) {
      return Binary(op, e);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    return SumAdd(a, b, -1);
  }
  // Mul
  Expr Mutate_(const Mul* op, const Expr& e) final {
    if (!EnableOpt(op->type)) {
      return Binary(op, e);
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
      return Binary(op, e);
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
      return Binary(op, e);
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
  // IntImm
  Expr Mutate_(const IntImm* op, const Expr& e) final {
    if (op->type != Int(32)) return e;
    auto it = cache_intimm_.find(op->value);
    if (it != cache_intimm_.end()) {
      return it->second;
    } else {
      cache_intimm_[op->value] = e;
      return e;
    }
  }
  // Div operator
  Expr Mutate_(const Div* op, const Expr& e) final {
    if (!EnableOpt(op->type)) {
      return Binary(op, e);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    if (is_const(a.value) && is_const(b.value)) {
      return ComputeExpr<Div>(a.value, b.value);
    } else if (is_const(b.value)) {
      return SumDivConst(a.AsSum(), b.value);
    } else {
      return Binary(op, e);
    }
  }
  // Mod operator
  Expr Mutate_(const Mod* op, const Expr& e) final {
    if (!EnableOpt(op->type)) {
      return Binary(op, e);
    }
    CacheEntry a = Produce(op->a);
    CacheEntry b = Produce(op->b);
    if (a.has_side_effect || b.has_side_effect) {
      return Binary_(op, e, a.value, b.value);
    }
    if (is_const(a.value) && is_const(b.value)) {
      return ComputeExpr<Mod>(a.value, b.value);
    } else if (is_const(b.value)) {
      return SumModConst(a.AsSum(), b.value);
    } else {
      return Binary(op, e);
    }
  }

  Expr Mutate_(const And* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<And>();
    if (is_one(op->a)) return op->b;
    if (is_one(op->b)) return op->a;
    return expr;
  }
  // Call
  Expr Mutate_(const Call* op, const Expr& e) final {
    if (!op->is_pure()) {
      stack_.back().has_side_effect = true;
    }
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Call>();
    if (op->is_intrinsic(Call::likely) && is_const(op->args[0])) {
      return op->args[0];
    } else {
      return expr;
    }
  }
  // For
  Stmt Mutate_(const For* op, const Stmt& s) {
    ++level_counter_;
    Var loop_var(op->loop_var.node_);
    this->SetRange(loop_var,
                   Range::make_by_min_extent(op->min, op->extent),
                   level_counter_);
    Stmt stmt = IRMutator::Mutate_(op, s);
    --level_counter_;
    return stmt;
  }
  // IfThenElse
  Stmt Mutate_(const IfThenElse* op, const Stmt& s) {
    Stmt stmt  = IRMutator::Mutate_(op, s);
    op = stmt.as<IfThenElse>();
    if (is_one(op->condition)) return op->then_case;
    return stmt;
  }
  // AttrStmt
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      ++level_counter_;
      IterVar iv(op->node.node_);
      CHECK_NE(iv->thread_tag.length(), 0U);
      if (!var_level_.count(iv->var.get())) {
        this->SetRange(iv->var,
                       Range::make_by_min_extent(0, op->value),
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
  template<typename T>
  Expr Binary(const T* op, Expr e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    BinaryExpr key{static_cast<int>(T::_type_info), a, b};
    auto it = cache_binary_.find(key);
    if (it != cache_binary_.end()) {
      return it->second;
    } else {
      Expr ret = Binary_(op, e, a, b);
      cache_binary_[key] = ret;
      return ret;
    }
  }
  // return entry
  CacheEntry ret_entry_;
  // internal information stack
  std::vector<StackEntry> stack_;
  // cache sum
  std::map<ComExpr, CacheEntry> cache_sum_;
  // cache of normal binary op
  std::map<BinaryExpr, Expr> cache_binary_;
  // cache of int constant
  std::unordered_map<int64_t, Expr> cache_intimm_;
  // range of each var
  std::unordered_map<const Variable*, IntSet> var_range_;
  // level of each var
  std::unordered_map<const Variable*, int> var_level_;
  // record history vars, to avoid false positive.
  std::vector<Var> var_rec_;
  // level counter
  int level_counter_{0};
  // get constant int value
  int64_t GetConstIntValue(const Expr& v) {
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
    return value;
  }
  // Detect if a = q * coeff + r, where r \in [0, coeff), coeff > 0
  // (in Euclidean division)
  // returns pair (q, r) if such detection is successful
  // returns empty vector otherwise.
  // Assumes that coeff is a constant integer
  std::vector<ComExpr> TryLinearEquation(const ComExpr& a,
                                         const Expr& coeff) {
    Type type = coeff.type();
    int64_t value = GetConstIntValue(coeff);
    CHECK_NE(value, 0);
    if (value < 0) return {};
    // Given that denominator (value variable) is positive, truncated division
    // (i.e., TVM's division semantics) is equivalent to Euclidean division if and only if
    // numerator is non-negative or numerator is divisible by denominator (i.e., value)
    IntSet numerator_int_set = EvalSet(Sum2Expr(a, type), var_range_);
    bool numerator_is_non_neg = numerator_int_set.can_prove_non_negative();
    // Try to separate terms of a into ones that can be proven to be
    // divisible by coeff and ones that are not
    // We will build q and r from divisible and non_divisible respectively
    auto divisible = make_node<ComExprNode>();
    auto non_divisible = make_node<ComExprNode>();
    if (a->base % value == 0) {
      divisible->base = a->base;
    } else {
      non_divisible->base = a->base;
    }
    for (const auto& e : a->elem) {
      if (e.scale % value == 0) {
        divisible->elem.push_back(e);
      } else {
        non_divisible->elem.push_back(e);
      }
    }
    bool non_divisible_is_simplified = false;
    int64_t div_result;
    Expr non_divisible_res = Sum2Expr(ComExpr(non_divisible), type);
    // if non_divisible part consists of only an integer and numerator is non-negative,
    // we can simply divide it by coeff
    if (is_const(non_divisible_res)) {
      int64_t non_divisible_const = GetConstIntValue(non_divisible_res);
      if (numerator_is_non_neg || non_divisible_const == 0) {
        non_divisible_is_simplified = true;
        // We need to do an Euclidean division here because (a*b + c)/b == a + c/b
        // holds true only if division is Euclidean
        div_result = HalideIR::Internal::div_imp(non_divisible_const , value);
      }
    } else {
      // If we can prove that non_divisible part lies within [0, coeff), then
      // non_divisible itself will be our r
      IntSet non_divisible_set = EvalSet(non_divisible_res, var_range_);
      if (non_divisible_set.min().type() == type &&
          non_divisible_set.max().type() == type) {
        if ( (non_divisible_set.is_single_point() &&
              can_prove(non_divisible_set.point_value() == 0)) ||
             (numerator_is_non_neg &&
              can_prove(non_divisible_set.min() >= make_zero(type)) &&
              can_prove(non_divisible_set.max() < coeff)) ) {
          non_divisible_is_simplified = true;
          div_result = 0;
        }
      }
    }
    if (non_divisible_is_simplified) {
      non_divisible->base -= div_result * value;
      divisible->base /= value;
      divisible->base += div_result;
      for (auto& e : divisible->elem) {
        e.scale /= value;
      }
      return {ComExpr(divisible), ComExpr(non_divisible)};
    } else {
      return {};
    }
  }
  // subroutine to do produce a % v
  Expr SumModConst(ComExpr a, Expr v) {
    std::vector<ComExpr> pair = TryLinearEquation(a, v);
    if (pair.size() == 0) {
      int64_t value = GetConstIntValue(v);
      auto n = make_node<ComExprNode>();
      // FIXME(derisavi) : The following can be done only for Euclidean division/mod.
      //  Therefore, it's only valid when truncated division/mod is equivalent to Euclidean one,
      //  that is, if and only if a and v are
      //  both negative or both positive or a is divisible by v.
      //  Extend the code to handle cases where the above condition is not satisfied, i.e.,
      //  a and v are of different signs and a is not divisible by v.
      n->base = a->base % value;
      for (auto e : a->elem) {
        if (e.scale % value == 0) continue;
        e.scale = e.scale % value;
        n->elem.push_back(e);
      }
      Expr ret = Sum2Expr(ComExpr(n), v.type()) % v;
      if (const Mod* mod = ret.as<Mod>()) {
        return Binary(mod, ret);
      } else {
        // Sometimes the result is a constant, this may happen when value is -1
        CHECK(is_const(ret)) << "CanonicalSimplify: "
          << Sum2Expr(ComExpr(n), v.type()) << " % " << v << " is " << ret
          << " which is neither Mod, nor a constant";
        return ret;
      }
    }
    ret_entry_.sum = pair[1];
    ret_entry_.max_level = stack_.back().max_level;
    ret_entry_.has_side_effect = stack_.back().has_side_effect;
    auto it = cache_sum_.find(ret_entry_.sum);
    if (it != cache_sum_.end()) {
      ret_entry_ = it->second;
    } else {
      ret_entry_.value = Sum2Expr(ret_entry_.sum, v.type());
      cache_sum_[ret_entry_.sum] = ret_entry_;
    }
    return ret_entry_.value;
  }
  // subroutine to do produce a % v
  Expr SumDivConst(ComExpr a, Expr v) {
    std::vector<ComExpr> pair = TryLinearEquation(a, v);
    if (pair.size() == 0) {
      Expr ret = Sum2Expr(a, v.type()) / v;
      return Binary(ret.as<Div>(), ret);
    }
    ret_entry_.sum = pair[0];
    ret_entry_.max_level = stack_.back().max_level;
    ret_entry_.has_side_effect = stack_.back().has_side_effect;
    auto it = cache_sum_.find(ret_entry_.sum);
    if (it != cache_sum_.end()) {
      ret_entry_ = it->second;
    } else {
      ret_entry_.value = Sum2Expr(ret_entry_.sum, v.type());
      cache_sum_[ret_entry_.sum] = ret_entry_;
    }
    return ret_entry_.value;
  }
  // subroutine to do produce
  Expr SumMulConst(ComExpr a, Expr v) {
    int64_t value = GetConstIntValue(v);
    if (value == 0) {
      return make_zero(v.type());
    }
    auto vsum = make_node<ComExprNode>(*a.operator->());
    vsum->base *= value;
    for (auto& e : vsum->elem) {
      e.scale *= value;
    }
    ret_entry_.sum = ComExpr(vsum);
    ret_entry_.max_level = stack_.back().max_level;
    ret_entry_.has_side_effect = stack_.back().has_side_effect;
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
    auto n = make_node<ComExprNode>();
    n->base = suma->base + sumb->base * bscale;
    // merge of suma and sumb;
    size_t i = 0, j = 0;
    while (i < suma->elem.size() && j < sumb->elem.size()) {
      const auto& a = suma->elem[i];
      const auto& b = sumb->elem[j];
      if (a.value.same_as(b.value) && a.level == b.level) {
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
    if (com->base > 0) {
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
    if (com->base < 0) {
      if (vsum.defined()) {
        vsum = Sub::make(vsum, make_const(t, -com->base));
      } else {
        vsum = make_const(t, com->base);
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

Canonical::Canonical(Map<Var, Range> vrange)
    : ptr_(std::make_shared<Internal>(vrange)) {}

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

Stmt CanonicalSimplify(Stmt stmt, Map<Var, Range> vrange) {
  return arith::Canonical(vrange).Simplify(stmt);
}

Expr CanonicalSimplify(Expr expr, Map<Var, Range> vrange) {
  return arith::Canonical(vrange).Simplify(expr);
}

template<typename T>
T Simplify_(T a, Map<Var, Range> vrange) {
  using namespace HalideIR::Internal;
  Scope<Interval> rscope;
  for (auto kv : vrange) {
    Range r = kv.second;
    rscope.push(
        kv.first.get(),
        Interval(r->min,
                 simplify(r->min + r->extent - make_const(r->min.type(), 1))));
  }
  return HalideIR::Internal::simplify(a, true, rscope);
}


/*!
 * \brief Simplify just the combiner of the given reduce node.
 *
 *  This function applies Simplify to the components of the top reduction's
 *  combiner, but not to the source or condition of the reduction.
 *  It also removes all components which are not used to
 *  compute the resulting value (the value_index-th value).
 *
 *  If \p expr is not a reduction node, it is left unchanged.
 *
 * \param expr The expression to be simplifed.
 * \return Simplified expression.
 */
Expr SimplifyCombiner(const Expr& expr, const Map<Var, Range>& vrange = Map<Var, Range>()) {
  const Reduce* op = expr.as<Reduce>();
  if (!op) {
    return expr;
  }

  // First simplify the results
  Array<Expr> simplified_result;
  for (const auto& res : op->combiner->result) {
    simplified_result.push_back(Simplify(res, vrange));
  }

  // Which components to keep
  std::vector<int> used(op->combiner->result.size(), false);

  // This function recursively marks the used components starting from
  // the index idx
  std::function<void(int)> mark_used;
  mark_used = [&used, &simplified_result, op, &mark_used](size_t idx) {
    // if the idx-th component was marked as used before, do nothing
    if (used[idx]) return;
    used[idx] = true;

    // check if the idx-th result expr uses some lhs or rhs variables
    // and recursively mark the corresponding components
    for (size_t i = 0; i < simplified_result.size(); ++i)
      if (!used[i]) {
        if (ExprUseVar(simplified_result[idx], op->combiner->lhs[i]) ||
            ExprUseVar(simplified_result[idx], op->combiner->rhs[i]))
          mark_used(i);
      }
  };

  // mark all used components starting from the value_index
  mark_used(op->value_index);

  // components which have side effects should also be preserved
  for (size_t i = 0; i < used.size(); ++i) {
    if (HasSideEffect(op->source[i]) || HasSideEffect(op->combiner->identity_element[i]) ||
        HasSideEffect(op->combiner->result[i])) {
      mark_used(i);
    }
  }

  int new_value_index = op->value_index;
  Array<Expr> new_result;
  Array<Expr> new_identity;
  Array<Var> new_lhs;
  Array<Var> new_rhs;
  Array<Expr> new_source;

  // new stuff is old stuff which is used
  for (size_t i = 0; i < used.size(); ++i) {
    if (used[i]) {
      // We simplify the result and identity, but not the source
      new_result.push_back(simplified_result[i]);
      new_identity.push_back(Simplify(op->combiner->identity_element[i], vrange));
      new_lhs.push_back(op->combiner->lhs[i]);
      new_rhs.push_back(op->combiner->rhs[i]);
      new_source.push_back(op->source[i]);
    } else if (static_cast<int>(i) < op->value_index) {
      // value_index should also be adjusted
      new_value_index--;
    }
  }

  CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
  return Reduce::make(new_combiner, new_source, op->axis, op->condition, new_value_index);
}

/*!
 * \brief Remove a single reduction over empty axis.
 *
 *  If \p e is a reduction node and its axis is empty, replace it with its source,
 *  otherwise return \p e unchanged.
 *
 * \param e The expression to be transformed.
 * \return The transformed expression.
 */
Expr RemoveEmptyReduction(const Expr& e) {
  const Reduce* r = e.as<Reduce>();
  if (r && r->axis.empty()) {
    // Note that here we assume that the identity element is indeed identity. Without this
    // assumption we would have to perform a single iteration of the loop, i.e. use
    // `(*r->combiner.get())(r->combiner->identity_element, r->source)[r->value_index]`
    // instead of `r->source[r->value_index]`. The former may be more difficult to simplify.
    return Select::make(r->condition,
                        r->source[r->value_index],
                        r->combiner->identity_element[r->value_index]);
  }
  return e;
}

Expr Simplify(Expr a, Map<Var, Range> vrange) {
  // We should not pass an expression having a non-HalideIR op to
  // Halide::Internal::simplify. Reduce op is the only such op at this time
  // and it only appears as the top op in an expression. So we strip it
  // first and send the sub-expressions to the simplifier.
  if (const Reduce* r = a.as<Reduce>()) {
    // If axis is empty, we can remove the reduce op completely.
    if (r->axis.empty())
      return Simplify_(RemoveEmptyReduction(a), vrange);

    // Simplify the combiner of the reduction
    a = SimplifyCombiner(a, vrange);
    r = a.as<Reduce>();

    // If axis is not empty then we add the information about ranges to vrange
    for (const IterVar& iv : r->axis) {
      if (vrange.count(iv->var)) {
        Range existing_range = vrange[iv->var];
        CHECK(Equal(existing_range->min, iv->dom->min) &&
              Equal(existing_range->extent, iv->dom->extent))
          << "Simplify was given vrange stating that the range of the reduction var "
          << iv << " is " << existing_range << ". This is probably a mistake.";
      }
      vrange.Set(iv->var, iv->dom);
    }

    Array<Expr> new_source;
    for (auto& e : r->source) {
      new_source.push_back(Simplify_(e, vrange));
    }
    Expr new_condition = Simplify_(r->condition, vrange);
    if (r->source.same_as(new_source) &&
        r->condition.same_as(new_condition)) {
      return a;
    } else {
      return Reduce::make(
              r->combiner, new_source, r->axis, new_condition, r->value_index);
    }
  }
  return Simplify_(a, vrange);
}

Stmt Simplify(Stmt a, Map<Var, Range> vrange) {
  return Simplify_(a, vrange);
}
}  // namespace ir
}  // namespace tvm
