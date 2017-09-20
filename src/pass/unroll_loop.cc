/*!
 *  Copyright (c) 2017 by Contributors
 *  Loop unrolling as in Halide pipeline.
 * \file unroll_loop.cc
 */
// Unrolls the loop as in Halide pipeline.
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

class LoopUnroller : public IRMutator {
 public:
  explicit LoopUnroller(int auto_max_step,
                        int auto_min_depth,
                        bool explicit_unroll)
      : auto_max_step_(auto_max_step),
        auto_min_depth_(auto_min_depth),
        explicit_unroll_(explicit_unroll) {
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
    bool auto_unroll = (op->for_type == ForType::Serial &&
                        value >= 0 && value <= auto_max_step_ &&
                        loop_depth_ >= auto_min_depth_);
    if (op->for_type == ForType::Unrolled) {
      CHECK_GE(value, 0)
          << "Cannot unroll non-constant loop";
      auto_unroll = true;
    }

    if (auto_unroll && explicit_unroll_) {
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
      ++loop_depth_;
      Stmt ret = this->Mutate(unrolled);
      --loop_depth_;
      return ret;
    } else {
      ++loop_depth_;
      Stmt ret = IRMutator::Mutate_(op, stmt);
      if (auto_unroll) {
        op = ret.as<For>();
        if (op->for_type != ForType::Unrolled) {
          ret = For::make(
              op->loop_var, op->min, op->extent,
              ForType::Unrolled, op->device_api, op->body);
        }
      }
      --loop_depth_;
      return ret;
    }
  }

 private:
  // maximum number of step to perform auto unroll.
  int auto_max_step_;
  int auto_min_depth_;
  bool explicit_unroll_;
  int loop_depth_{0};
};


Stmt UnrollLoop(Stmt stmt,
                int auto_max_step,
                int auto_min_depth,
                bool explicit_unroll) {
  Stmt ret = LoopUnroller(
      auto_max_step,
      auto_min_depth,
      explicit_unroll).Mutate(stmt);
  if (!ret.same_as(stmt)) {
    return ConvertSSA(ret);
  } else {
    return ret;
  }
}

}  // namespace ir
}  // namespace tvm
