/*!
 *  Copyright (c) 2019 by Contributors
 * \file rewrite_simplify.cc
 * \brief Rewrite-rule based simplification.
 */
// Acknowledgement: Most rewrite-rules are from Halide.
#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_mutator.h>
#include "const_fold.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace ir;

// macro for doing simple rewrite
#define TVM_TRY_REWRITE(SrcExpr, ResExpr)       \
  if ((SrcExpr).Match(ret)) {                   \
    return (ResExpr).Eval();                    \
  }

// macro for rewrite + recursively rewrite ResExpr
#define TVM_TRY_RECURSIVE_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {                       \
    return RecursiveRewrite((ResExpr).Eval());      \
  }

// macro rewrite only if CondExor is true after match.
#define TVM_TRY_REWRITE_IF(SrcExpr, ResExpr, CondExpr)  \
  if ((SrcExpr).Match(ret) && (CondExpr)) {             \
    return (ResExpr).Eval();                            \
  }

// macro rewrite + recursive_rewrite only if CondExor is true after match.
#define TVM_TRY_RECURSIVE_REWRITE_IF(SrcExpr, ResExpr, CondExpr)  \
  if ((SrcExpr).Match(ret) && (CondExpr)) {                       \
    return RecursiveRewrite((ResExpr).Eval());                    \
  }


// NOTE for developers:
//
// We mainly focus on index expression simplification.
// Besides the RewriteSimplifier, some cases can be better
// handled by CanonicalSimplifier.
//
class RewriteSimplifier::Impl : public IRMutator {
 public:
  explicit Impl(Analyzer* parent)
      : parent_(parent) {}

  void Update(const Var& var,
              const Expr& info,
              bool override) {
    if (!override) {
      CHECK(!var_map_.count(var));
    }
    var_map_[var] = info;
  }

  // Run simplification in post order
  Expr PostOrderSimplify(Expr expr, int max_iter = 2) {
    for (int i = 0; i < max_iter; ++i) {
      Expr new_expr = this->Mutate(expr);
      if (new_expr.same_as(expr)) return expr;
      expr = new_expr;
    }
    return expr;
  }

  Expr Mutate_(const Add* op, const Expr& self) final;
  Expr Mutate_(const Sub* op, const Expr& self) final;
  Expr Mutate_(const Mul* op, const Expr& self) final;
  Expr Mutate_(const Div* op, const Expr& self) final;
  Expr Mutate_(const Mod* op, const Expr& self) final;

 private:
  // reference to the main analyzer
  Analyzer* parent_;
  // counter to record recursive rewrite depth.
  int recur_depth_{0};
  // internal variable map
  std::unordered_map<Var, Expr, ExprHash, ExprEqual> var_map_;
  // maximum number of recursion allowed during a single pass.
  static const constexpr int kMaxRecurDepth = 5;
  // Whether x >= val
  bool CanProveGreaterEqual(const Expr& x, int64_t val) {
    return parent_->CanProveGreaterEqual(x, val);
  }
  // Whether x == val
  bool CanProveEqual(const Expr& x, int64_t val) {
    // TODO(tqchen) refer back to super-analyzer.
    Expr res = Mutate(x);
    if (const auto* ptr = res.as<ir::IntImm>()) {
      return ptr->value == val;
    }
    return false;
  }
  // Recursive rewrite x
  // we limit maximum depth of recursive rewrite allowed to
  // avoid infinite loop
  Expr RecursiveRewrite(const Expr& x) {
    if (recur_depth_ >= kMaxRecurDepth) return x;
    ++recur_depth_;
    Expr res = Mutate(x);
    --recur_depth_;
    return res;
  }

  template<typename TA>
  PConstWithTypeLike<TA> ZeroWithTypeLike(const Pattern<TA>& pattern) {
    return PConstWithTypeLike<TA>(pattern.derived(), 0);
  }
};

Expr RewriteSimplifier::Impl::
Mutate_(const Add* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Add>();
  Expr const_res = TryConstFold<Add>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) + ramp(b2, s2, lanes),
                    ramp(b1 + b2, s1 + s2, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) + broadcast(x, lanes),
                    ramp(b1 + x, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) + ramp(b1, s1, lanes),
                    ramp(x + b1, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) + broadcast(y, lanes),
                    broadcast(x + y, lanes));
  }

  if (IsIndexType(op->type)) {
    // Index rules
    // cancelation rules
    TVM_TRY_REWRITE((x - y) + y, x);
    TVM_TRY_REWRITE(x + (y - x), y);

    TVM_TRY_REWRITE((x - y) + (y - z), x - z);
    TVM_TRY_REWRITE((x - y) + (z - x), z - y);

    TVM_TRY_REWRITE(min(x, y - z) + z, min(x + z, y));
    TVM_TRY_REWRITE(min(x - z, y) + z, min(x, y + z));
    TVM_TRY_REWRITE(max(x, y - z) + z, max(x + z, y));
    TVM_TRY_REWRITE(max(x - z, y) + z, max(x, y + z));
    TVM_TRY_REWRITE(max(x, y) + min(x, y), x + y);
    TVM_TRY_REWRITE(min(x, y) + max(x, y), x + y);
    TVM_TRY_REWRITE(max(x, y) + min(y, x), x + y);
    TVM_TRY_REWRITE(min(x, y) + max(y, x), x + y);

    TVM_TRY_REWRITE_IF(min(x, y + c1) + c2, min(x + c2, y),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(min(x + c1, y) + c2, min(x, y + c2),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x, y + c1) + c2, max(x + c2, y),
                       c1.Eval()->value == -c2.Eval()->value);
    TVM_TRY_REWRITE_IF(max(x + c1, y) + c2, max(x, y + c2),
                       c1.Eval()->value == -c2.Eval()->value);

    // constant folding
    // NOTE: canonicalization might better at this.
    TVM_TRY_REWRITE((x + c1) + c2, x + (c1 + c2));

    // mul co-efficient folding
    TVM_TRY_REWRITE(x + x, x * 2);
    TVM_TRY_REWRITE(x * y + x, x * (y + 1));
    TVM_TRY_REWRITE(y * x + x, x * (y + 1));
    TVM_TRY_REWRITE(x + y * x, x * (1 + y));
    TVM_TRY_REWRITE(x + x * y, x * (1 + y));
    TVM_TRY_REWRITE(x * y + x * z, x * (y + z));
    TVM_TRY_REWRITE(y * x + x * z, x * (y + z));
    TVM_TRY_REWRITE(x * y + z * x, x * (y + z));
    TVM_TRY_REWRITE(y * x + z * x, x * (y + z));

    // modular-div simplification
    // Always pre-condition on positive integer domain
    TVM_TRY_REWRITE_IF(
        (x / c1) * c1 + x % c1, x,
        CanProveGreaterEqual(x.Eval(), 0) && c1.Eval()->value > 0);

    // canonicalization rule
    // will try rewrite again after canonicalization.
    TVM_TRY_RECURSIVE_REWRITE(x + (c1 - y), (x - y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + c1 + y, (x + y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x + (c1 + y), (x + y) + c1);
    TVM_TRY_RECURSIVE_REWRITE((y % c1) + x * c1, x * c1 + (y % c1));
  }

  // condition rules.
  TVM_TRY_REWRITE(select(x, b1, b2) + select(x, s1, s2),
                  select(x, b1 + s1, b2 + s2));
  // default value
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Sub* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Sub>();
  Expr const_res = TryConstFold<Sub>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) - ramp(b2, s2, lanes),
                    ramp(b1 - b2, s1 - s2, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) - broadcast(x, lanes),
                    ramp(b1 - x, s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) - ramp(b1, s1, lanes),
                    ramp(x - b1, 0 - s1, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) - broadcast(y, lanes),
                    broadcast(x - y, lanes));
  }

  if (IsIndexType(op->type)) {
    // Index rules
    // cancelation rules
    TVM_TRY_REWRITE((x + y) - y, x);
    TVM_TRY_REWRITE((x + y) - x, y);
    TVM_TRY_REWRITE(x - (y + x), 0 - y);
    TVM_TRY_REWRITE(x - (x + y), 0 - y);

    TVM_TRY_REWRITE(min(x, y) - x, min(0, y - x));
    TVM_TRY_REWRITE(min(x, y) - y, min(x - y, 0));
    TVM_TRY_REWRITE(max(x, y) - x, max(0, y - x));
    TVM_TRY_REWRITE(max(x, y) - y, max(x - y, 0));

    TVM_TRY_REWRITE(x - max(x, y), min(0, x - y));
    TVM_TRY_REWRITE(y - max(x, y), min(y - x, 0));
    TVM_TRY_REWRITE(x - min(x, y), max(0, x - y));
    TVM_TRY_REWRITE(y - min(x, y), max(y - x, 0));

    // mul co-efficient folding
    TVM_TRY_REWRITE(x - x, ZeroWithTypeLike(x));
    TVM_TRY_REWRITE(x * y - x, x * (y - 1));
    TVM_TRY_REWRITE(y * x - x, x * (y - 1));
    TVM_TRY_REWRITE(x - y * x, x * (1 - y));
    TVM_TRY_REWRITE(x - x * y, x * (1 - y));
    TVM_TRY_REWRITE(x * y - x * z, x * (y - z));
    TVM_TRY_REWRITE(y * x - x * z, x * (y - z));
    TVM_TRY_REWRITE(x * y - z * x, x * (y - z));
    TVM_TRY_REWRITE(y * x - z * x, x * (y - z));

    // constant cancelation
    TVM_TRY_REWRITE((x + c1) - c2, x + (c1 - c2));
    TVM_TRY_REWRITE((c1 - x) - (c2 - y), (y - x) + (c1 - c2));

    // cancelization rule involving 4 operands
    TVM_TRY_REWRITE((x + y) - (x + z), y - z);
    TVM_TRY_REWRITE((x + y) - (z + x), y - z);
    TVM_TRY_REWRITE((y + x) - (z + x), y - z);
    TVM_TRY_REWRITE((y + x) - (x + z), y - z);

    TVM_TRY_REWRITE(min(x + y, z) - x,  min(y, z - x));
    TVM_TRY_REWRITE(min(y + x, z) - x,  min(y, z - x));
    TVM_TRY_REWRITE(min(z, x + y) - x,  min(z - x, y));
    TVM_TRY_REWRITE(min(z, y + x) - x,  min(z - x, y));

    TVM_TRY_REWRITE(x - min(x + y, z),  max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(y + x, z),  max(0 - y, x - z));
    TVM_TRY_REWRITE(x - min(z, x + y),  max(x - z, 0 - y));
    TVM_TRY_REWRITE(x - min(z, y + x),  max(x - z, 0 - y));

    TVM_TRY_REWRITE(min(x, y) - min(y, x), ZeroWithTypeLike(x));
    TVM_TRY_REWRITE(max(x, y) - max(y, x), ZeroWithTypeLike(x));

    TVM_TRY_REWRITE_IF(min(b1, b2) - min(s1, s2), b1 - s1,
                       CanProveEqual(((b1 - s1) - (b2 - s2)).Eval(), 0));

    TVM_TRY_REWRITE_IF(min(b1, b2) - min(s1, s2), b1 - s2,
                       CanProveEqual(((b1 - s2) - (b2 - s1)).Eval(), 0));
    TVM_TRY_REWRITE_IF(max(b1, b2) - max(s1, s2), b1 - s1,
                       CanProveEqual(((b1 - s1) - (b2 - s2)).Eval(), 0));
    TVM_TRY_REWRITE_IF(max(b1, b2) - max(s1, s2), b1 - s2,
                       CanProveEqual(((b1 - s2) - (b2 - s1)).Eval(), 0));

    // modular-div simplification
    // Always pre-condition on positive integer domain
    TVM_TRY_REWRITE_IF(x - (x / c1) * c1, x % c1,
                       CanProveGreaterEqual(x.Eval(), 0) && c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF((x / c1) * c1 - x, 0 - (x % c1),
                       CanProveGreaterEqual(x.Eval(), 0) && c1.Eval()->value > 0);
    TVM_TRY_REWRITE_IF((x + c1) / c3  - (x + c2) / c3,
                       ((x + (c1 % c3)) % c3 + (c1 - c2)) / c3,
                       CanProveGreaterEqual(x.Eval(), -c2.Eval()->value) &&
                       c1.Eval()->value >= c2.Eval()->value &&
                       c3.Eval()->value > 0);
    TVM_TRY_REWRITE_IF((x + c1) / c3  - x / c3,
                       ((x + (c1 % c3)) % c3 + c1) / c3,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       c1.Eval()->value >= 0 &&
                       c3.Eval()->value > 0);
    // canonicalization rule
    // will try rewrite again after canonicalization.
    TVM_TRY_REWRITE(x - c1, x + (0 - c1));
    TVM_TRY_RECURSIVE_REWRITE((x + c1) - y, (x - y) + c1);
    TVM_TRY_RECURSIVE_REWRITE(x - (y - z), (x + z) - y);
    TVM_TRY_RECURSIVE_REWRITE(x - y * c1, x + y * (0 - c1));
  }

  // condition rules.
  TVM_TRY_REWRITE(select(x, b1, b2) - select(x, s1, s2),
                  select(x, b1 - s1, b2 - s2));
  TVM_TRY_REWRITE(select(x, y, z) - z,
                  select(x, y - z, ZeroWithTypeLike(z)));
  TVM_TRY_REWRITE(select(x, y, z) - y,
                  select(x, ZeroWithTypeLike(y), z - y));
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Mul* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Mul>();
  Expr const_res = TryConstFold<Mul>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1, b2, s1, s2;
  // Pattern var match IntImm
  PVar<Integer> c1, c2;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;
  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) * broadcast(y, lanes),
                    broadcast(x * y, lanes));
    TVM_TRY_REWRITE(ramp(b1, s1, lanes) * broadcast(x, lanes),
                    ramp(b1 * x, s1 * x, lanes));
    TVM_TRY_REWRITE(broadcast(x, lanes) * ramp(b1, s1, lanes),
                    ramp(b1 * x, s1 * x, lanes));
  }

  if (IsIndexType(op->type)) {
    // constant simplification rule
    TVM_TRY_REWRITE((x + c1) * c2, x * c2 + c1 * c2);
    TVM_TRY_REWRITE((x * c1) * c2, x * (c1 * c2));
    TVM_TRY_REWRITE(min(x, y) * max(x, y), x * y);
    TVM_TRY_REWRITE(max(x, y) * min(x, y), x * y);

    // canonicalization
    TVM_TRY_RECURSIVE_REWRITE(x * (c1 * y), (x * y) * c1);
    TVM_TRY_RECURSIVE_REWRITE_IF(
        (x - y) * c1, (y - x) * (0 - c1),
        c1.Eval()->value < 0);
  }
  return ret;
}

Expr RewriteSimplifier::Impl::
Mutate_(const Div* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Div>();
  Expr const_res = TryConstFold<Div>(op->a, op->b);
  if (const_res.defined()) return const_res;
  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) / broadcast(y, lanes),
                    broadcast(x / y, lanes));
    // ramp / bcast
    if ((ramp(b1, c1, lanes) / broadcast(c2, lanes)).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return ramp(b1 / c2, c1 / c2, lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      if (CanProveGreaterEqual(b1.Eval(), 0)) {
        ModularSet bmod = parent_->modular_set(b1.Eval());
        int64_t ramp_min = bmod->base / c2val;
        int64_t ramp_max = (bmod->base + (lanes.Eval() - 1) * c1val) / c2val;
        if (bmod->coeff % c2val == 0 && ramp_min == ramp_max) {
          return broadcast(b1 / c2, lanes).Eval();
        }
      }
    }
  }

  if (IsIndexType(op->type)) {
    // Be-aware of the division rules:
    // We adopt the default C division uses truncation instead of floordiv.
    // This means most rules need to check non-negativeness of the operands.

    // while it is always true for trunc div
    // restrict to common case(positive div)
    TVM_TRY_REWRITE_IF((x / c1) / c2, x / (c1 * c2),
                       c1.Eval()->value > 0 && c2.Eval()->value > 0);

    TVM_TRY_REWRITE_IF((x / c1 + c2) / c3, (x + c1 * c2) / (c1 * c3),
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value >= 0 &&
                       c3.Eval()->value > 0 &&
                       CanProveGreaterEqual(x.Eval(), 0));

    if (((x * c1) / c2).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val > 0 && c2val > 0) {
        if (c1val % c2val == 0) return (x * (c1 / c2)).Eval();
        if (c2val % c1val == 0) return (x / (c2 / c1)).Eval();
      }
    }

    // Rules involving 2-operands.
    TVM_TRY_REWRITE_IF((x * c1 + y) / c2, x * (c1 / c2) + y / c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(min(x * c1, y) / c2, min(x * (c1 / c2), y / c2),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(max(x * c1, y) / c2, max(x * (c1 / c2), y / c2),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF((y + x * c1) / c2, y / c2 + x * (c1 / c2),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(min(y, x * c1) / c2, min(y / c2, x * (c1 / c2)),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(max(y, x * c1) / c2, max(y / c2, x * (c1 / c2)),
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    // Rules involving 3-operands.
    TVM_TRY_REWRITE_IF((x * c1 + y + z) / c2, x * (c1 / c2) + (y + z)/ c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF((x * c1 - y + z) / c2, x * (c1 / c2) + (z - y)/ c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((z - y).Eval(), 0));

    TVM_TRY_REWRITE_IF((x * c1 + y - z) / c2, x * (c1 / c2) + (y - z)/ c2,
                       c1.Eval()->value >= 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y - z).Eval(), 0));

    TVM_TRY_REWRITE_IF((y + x * c1 + z) / c2, x * (c1 / c2) + (y + z) / c2,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF((x + c1) / c2, x / c2 + c1 / c2,
                       c1.Eval()->value > 0 &&
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF((x + y) / x, y / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF((y + x) / x, y / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF(((x + y) + z) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF(((y + x) + z) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF((y + (z + x)) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));
    TVM_TRY_REWRITE_IF((y + (x + z)) / x, (y + z) / x + 1,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual((y + z).Eval(), 0));

    TVM_TRY_REWRITE_IF((x * y) / y, x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));
    TVM_TRY_REWRITE_IF((y * x) / y, x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF((x * z + y) / z, x + y / z,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF((z * x + y) / z, x + y / z,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF((y + x * z) / z, y / z + x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
    TVM_TRY_REWRITE_IF((y + z * x) / z, y / z + x,
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0) &&
                       CanProveGreaterEqual(z.Eval(), 0));
  }
  return ret;
}


Expr RewriteSimplifier::Impl::
Mutate_(const Mod* op, const Expr& self) {
  Expr ret = IRMutator::Mutate_(op, self);
  op = ret.as<Mod>();
  Expr const_res = TryConstFold<Mod>(op->a, op->b);
  if (const_res.defined()) return const_res;

  // Pattern var to match any expression
  PVar<Expr> x, y, z, b1;
  // Pattern var match IntImm
  PVar<Integer> c1, c2, c3;
  // Pattern var for lanes in broadcast and ramp
  PVar<int> lanes;

  // Vector rules
  if (op->type.lanes() != 1) {
    TVM_TRY_REWRITE(broadcast(x, lanes) % broadcast(y, lanes),
                    broadcast(x % y, lanes));

    // ramp % bcast
    if ((ramp(b1, c1, lanes) % broadcast(c2, lanes)).Match(ret)) {
      int64_t c1val = c1.Eval()->value;
      int64_t c2val = c2.Eval()->value;
      if (c1val % c2val == 0) {
        return broadcast(b1 % c2, lanes).Eval();
      }
      // If all possible indices in ramp are the same.
      if (CanProveGreaterEqual(b1.Eval(), 0)) {
        ModularSet bmod = parent_->modular_set(b1.Eval());
        int64_t ramp_min = bmod->base / c2val;
        int64_t ramp_max = (bmod->base + (lanes.Eval() - 1) * c1val) / c2val;
        if (bmod->coeff % c2val == 0) {
          if (ramp_min == ramp_max) {
            return ramp(bmod->base % c2, c1, lanes).Eval();
          } else {
            return (ramp(bmod->base % c2, c1, lanes) % broadcast(c2, lanes)).Eval();
          }
        }
      }
    }
  }

  if (IsIndexType(op->type)) {
    // Be-aware of the division rules:
    // We adopt the default C division uses truncation instead of floordiv.
    // This means most rules need to check non-negativeness of the operands.
    TVM_TRY_REWRITE_IF((x * c1) % c2, ZeroWithTypeLike(x),
                       c2.Eval()->value != 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0);

    TVM_TRY_REWRITE_IF((x * c1 + y) % c2, y % c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(y.Eval(), 0));

    TVM_TRY_REWRITE_IF((x + c1) % c2, x % c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0));

    TVM_TRY_REWRITE_IF((x + y * c1) % c2, x % c2,
                       c2.Eval()->value > 0 &&
                       c1.Eval()->value % c2.Eval()->value == 0 &&
                       CanProveGreaterEqual(x.Eval(), 0) &&
                       CanProveGreaterEqual(y.Eval(), 0));

    // try modular analysis
    if ((x % c1).Match(ret)) {
      ModularSet mod = parent_->modular_set(x.Eval());
      int64_t c1val = c1.Eval()->value;
      if (mod->coeff % c1val == 0 &&
          CanProveGreaterEqual(x.Eval(), 0)) {
        return (mod->base % c1).Eval();
      }
    }
  }
  return ret;
}


Expr RewriteSimplifier::operator()(const Expr& expr) {
  return impl_->PostOrderSimplify(expr);
}

void RewriteSimplifier::Update(const Var& var,
                               const Expr& info,
                               bool override) {
  impl_->Update(var, info, override);
}


RewriteSimplifier::RewriteSimplifier(Analyzer* parent)
    : impl_(new Impl(parent)) {
}

RewriteSimplifier::~RewriteSimplifier() {
  delete impl_;
}

}  // namespace arith
}  // namespace tvm
