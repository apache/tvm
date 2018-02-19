/*!
 *  Copyright (c) 2017 by Contributors
 * \file unsafe_select_rewrite.cc
 * \brief Rewrite uinsafe select expression.
 */
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {


// For now, rewrite unsafe select expression to if_then_else
// TODO(tqchen) pattern matching to support masked load
class UnsafeExprDetector : public ExprFunctor<bool(const Expr& n)> {
 public:
  // select itself is always considered safe if condition is safe
  // Because we will issue guard to make sure it is.
  bool VisitExpr_(const Select* op) {
    return VisitExpr(op->condition);
  }
  bool VisitExpr_(const Call* op) {
    if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
      return VisitExpr(op->args[0]);
    } else if (op->is_intrinsic(intrinsic::tvm_address_of)) {
      const Load* l = op->args[0].as<Load>();
      return this->VisitExpr(l->index);
    } else if (op->is_pure()) {
      for (Expr e : op->args) {
        if (VisitExpr(e)) return true;
      }
      return false;
    } else {
      return true;
    }
  }
  bool VisitExpr_(const Load* op) {
    // Load is considered unsafe.
    return true;
  }
  bool VisitExpr_(const Add* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Sub* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Mul* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Div* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Mod* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Min* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Max* op) final { return BinaryOp(op); }
  bool VisitExpr_(const EQ* op) final { return BinaryOp(op); }
  bool VisitExpr_(const NE* op) final { return BinaryOp(op); }
  bool VisitExpr_(const LT* op) final { return BinaryOp(op); }
  bool VisitExpr_(const LE* op) final { return BinaryOp(op); }
  bool VisitExpr_(const GT* op) final { return BinaryOp(op); }
  bool VisitExpr_(const GE* op) final { return BinaryOp(op); }
  bool VisitExpr_(const And* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Or* op) final { return BinaryOp(op); }
  bool VisitExpr_(const Not* op) final {
    return VisitExpr(op->a);
  }
  bool VisitExpr_(const Let* op) final {
    return VisitExpr(op->body) || VisitExpr(op->value);
  }
  bool VisitExpr_(const Cast* op) final {
    return VisitExpr(op->value);
  }
  bool VisitExpr_(const Broadcast* op) final {
    return VisitExpr(op->value);
  }
  bool VisitExpr_(const Ramp* op) final {
    return VisitExpr(op->base) && VisitExpr(op->stride);
  }
  bool VisitExpr_(const Shuffle* op) final {
    for (Expr e : op->vectors) {
      if (VisitExpr(e)) return true;
    }
    return false;
  }
  bool VisitExpr_(const Variable* op) final { return false; }
  bool VisitExpr_(const UIntImm* op) final { return false; }
  bool VisitExpr_(const IntImm* op) final { return false; }
  bool VisitExpr_(const FloatImm* op) final { return false; }
  bool VisitExpr_(const StringImm* op) final { return false; }

 private:
  template<typename T>
  bool BinaryOp(const T* op) {
    return VisitExpr(op->a) || VisitExpr(op->b);
  }
};

class UnsafeSelectRewriter : public IRMutator {
 public:
  Expr Mutate_(const Select* op, const Expr& e) {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Select>();
    UnsafeExprDetector unsafe;
    if (unsafe.VisitExpr(op->true_value) ||
        unsafe.VisitExpr(op->false_value)) {
      return Call::make(
          op->type,
          intrinsic::tvm_if_then_else,
          {op->condition, op->true_value, op->false_value},
          Call::Intrinsic);
    } else {
      return expr;
    }
  }
};

Stmt RewriteUnsafeSelect(Stmt stmt) {
  return UnsafeSelectRewriter().Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
