/*!
 *  Copyright (c) 2016 by Contributors
 *  SSA related checks and pass.
 * \file ssa.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic//compute_expr.h"

namespace tvm {
namespace ir {

class LoopUnroller : public IRMutator {
 public:
  explicit LoopUnroller(int max_auto_step)
      : max_auto_step_(max_auto_step) {
  }

  Stmt Mutate_(const For* op, const Stmt& s) {
    Stmt stmt = s;
    // constant folding.
    Expr extent = ir::Simplify(op->extent);
    const IntImm* v1 = extent.as<IntImm>();
    const UIntImm* v2 = extent.as<UIntImm>();
    int value = -1;
    if (v1 != nullptr) {
      value = static_cast<int>(v1->value);
    }
    if (v2 != nullptr) {
      value = static_cast<int>(v2->value);
    }
    bool allow_unroll = value >= 0 && value <= max_auto_step_;
    if (op->for_type == ForType::Unrolled) {
      CHECK_GE(value, 0)
          << "Cannot unroll non-constant loop";
      allow_unroll = true;
    }

    if (allow_unroll) {
      using arith::ComputeExpr;
      if (value == 0) return Evaluate::make(0);
      Stmt body = op->body;
      Map<Var, Expr> vmap;
      Stmt unrolled;
      for (int i = 0; i < value; ++i) {
        Var lv(op->loop_var.node_);
        vmap.Set(lv,
                 ComputeExpr<Add>(
                     op->min, make_const(op->loop_var.type(), i)));
        Stmt step = Substitute(body, vmap);
        if (unrolled.defined()) {
          unrolled = Block::make(unrolled, step);
        } else {
          unrolled = step;
        }
      }
      return this->Mutate(unrolled);
    } else {
      return IRMutator::Mutate_(op, stmt);
    }
  }

 private:
  int max_auto_step_;
};


Stmt UnrollLoop(Stmt stmt, int max_auto_step) {
  Stmt ret = LoopUnroller(max_auto_step).Mutate(stmt);
  return ConvertSSA(ret);
}

}  // namespace ir
}  // namespace tvm
