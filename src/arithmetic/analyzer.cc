/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/arithmetic/analyzer.cc
 */
#include <tvm/ir.h>
#include <tvm/arithmetic.h>

namespace tvm {
namespace arith {

Analyzer::Analyzer()
    : const_int_bound(this),
      modular_set(this),
      rewrite_simplify(this) {
}

void Analyzer::Bind(const VarExpr& v, const Expr& expr) {
  Var var(v.node_);
  this->const_int_bound.Update(var, this->const_int_bound(expr));
  this->modular_set.Update(var, this->modular_set(expr));
  this->rewrite_simplify.Update(var, this->rewrite_simplify(expr));
}

void Analyzer::Bind(const VarExpr& v, const Range& range) {
  Var var(v.node_);
  this->const_int_bound.Bind(var, range);
  // skip modular_set
  // skip rewrite simplify
}

ConstraintContext::ConstraintContext(Analyzer* analyzer, const Expr& constraint) {
  // entering the scope.
  auto f0 = analyzer->const_int_bound.EnterConstraint(constraint);
  auto f1 = analyzer->modular_set.EnterConstraint(constraint);
  // recovery function.
  exit_ = [f0, f1]() {
    if (f1 != nullptr) f1();
    if (f0 != nullptr) f0();
  };
}

bool Analyzer::CanProveGreaterEqual(const Expr& expr, int64_t lower_bound) {
  if (const auto* ptr = expr.as<ir::IntImm>()) {
    return ptr->value > lower_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->min_value >= lower_bound) return true;
  return false;
}
}  // namespace arith
}  // namespace tvm
