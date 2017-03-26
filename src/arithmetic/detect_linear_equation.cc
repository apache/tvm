/*!
 *  Copyright (c) 2017 by Contributors
 * \file bound_deducer.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/arithmetic.h>
#include "./compute_expr.h"

namespace tvm {
namespace arith {

using namespace ir;

// Linear equation, the components can be undefined.
struct LinearEqEntry {
  Expr base;
  Expr coeff;
};

class LinearEqDetector
    : public ExprFunctor<LinearEqEntry(const Expr&, const Expr &)> {
 public:
  explicit LinearEqDetector(Var var)
      : var_(var) {}

  Array<Expr> Detect(const Expr& e) {
    LinearEqEntry ret = VisitExpr(e, e);
    if (fail_) return Array<Expr>();
    if (!ret.base.defined()) {
      ret.base = make_zero(var_.type());
    }
    if (!ret.coeff.defined()) {
      ret.coeff = make_zero(var_.type());
    }
    return Array<Expr>{ret.base, ret.coeff};
  }

  LinearEqEntry VisitExpr_(const Add* op, const Expr& e) final {
    if (fail_) return LinearEqEntry();
    LinearEqEntry a = VisitExpr(op->a, op->a);
    LinearEqEntry b = VisitExpr(op->b, op->b);
    LinearEqEntry ret;
    ret.base = AddCombine(a.base, b.base);
    ret.coeff = AddCombine(a.coeff, b.coeff);
    return ret;
  }
  LinearEqEntry VisitExpr_(const Mul* op, const Expr& e) final {
    if (fail_) return LinearEqEntry();
    LinearEqEntry a = VisitExpr(op->a, op->a);
    LinearEqEntry b = VisitExpr(op->b, op->b);
    if (a.coeff.defined()) {
      std::swap(a, b);
    }
    if (a.coeff.defined()) {
      fail_ = true;
      return LinearEqEntry();
    }
    LinearEqEntry ret;
    ret.base = MulCombine(a.base, b.base);
    ret.coeff = MulCombine(a.base, b.coeff);
    return ret;
  }
  LinearEqEntry VisitExpr_(const Variable* op, const Expr& e) final {
    LinearEqEntry ret;
    if (op == var_.get()) {
      ret.coeff = make_const(op->type, 1);
    } else {
      ret.base = e;
    }
    return ret;
  }
  LinearEqEntry VisitExprDefault_(const Node* op, const Expr& e) final {
    if (fail_) return LinearEqEntry();
    if (ExprUseVar(e, var_)) {
      fail_ = true;
      return LinearEqEntry();
    } else {
      LinearEqEntry ret;
      ret.base = e;
      return ret;
    }
  }

 private:
  Var var_;
  bool fail_{false};
  // Combine by add
  Expr AddCombine(Expr a, Expr b) {
    if (!a.defined()) return b;
    if (!b.defined()) return a;
    return ComputeExpr<Add>(a, b);
  }
  Expr MulCombine(Expr a, Expr b) {
    if (!a.defined()) return a;
    if (!b.defined()) return b;
    return ComputeExpr<Mul>(a, b);
  }
};

Array<Expr> DetectLinearEquation(Expr e, Var var) {
  return LinearEqDetector(var).Detect(e);
}

}  // namespace arith
}  // namespace tvm
