/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file renormalize_split_pattern.cc
 * \brief Renormalize the split pattern from floordiv(floormod()) to floormod(floordiv())
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

using namespace arith;

// macro for doing simple rewrite
#define TRY_REWRITE(SrcExpr, ResExpr) \
  if ((SrcExpr).Match(ret)) {         \
    return (ResExpr).Eval();          \
  }

// macro rewrite + recursive_rewrite only if CondExpr is true after match.
#define TRY_RECURSIVE_REWRITE_IF(SrcExpr, ResExpr, CondExpr) \
  if ((SrcExpr).Match(ret) && (CondExpr)) {                  \
    return RecursiveRewrite((ResExpr).Eval());               \
  }

class SplitPatternReNormalizer : public IRMutatorWithAnalyzer {
 public:
  explicit SplitPatternReNormalizer(Analyzer* analyzer) : IRMutatorWithAnalyzer(analyzer) {}

  using IRMutatorWithAnalyzer::VisitExpr_;

  PrimExpr VisitExpr_(const FloorDivNode* op) final {
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    PrimExpr ret = floordiv(a, b);
    // Pattern var to match any expression
    PVar<PrimExpr> x, y, z;
    // Pattern var match IntImm
    PVar<IntImm> c1, c2, c3;
    // Pattern var for lanes in broadcast and ramp
    PVar<PrimExpr> lanes;

    // floordiv(floormod(x, c1 * c2), c2) = floormod(floordiv(x, c2), c1)
    TRY_RECURSIVE_REWRITE_IF(floordiv(floormod(x, c3), c2),
                             floormod(floordiv(x, c2), floordiv(c3, c2)),
                             c3.Eval()->value % c2.Eval()->value == 0);
    TRY_RECURSIVE_REWRITE_IF(
        floordiv(floormod(x, broadcast(c3, lanes)), broadcast(c2, lanes)),
        floormod(floordiv(x, broadcast(c2, lanes)), broadcast(floordiv(c3, c2), lanes)),
        c3.Eval()->value % c2.Eval()->value == 0);

    // floordiv(x*c1*c3 + y, c2*c3) = floordiv(x*c1 + floordiv(y, c3), c2)
    if ((floordiv(x * c1 + y, c2)).Match(ret)) {
      int64_t c1_val = c1.Eval()->value;
      int64_t c2_val = c2.Eval()->value;
      if (c1_val > 0 && c2_val > 0) {
        int64_t c3 = ZeroAwareGCD(c1_val, c2_val);
        if (c3 > 1) {
          IntImm c1_div = IntImm(c1.Eval().dtype(), c1_val / c3);
          IntImm c2_div = IntImm(c2.Eval().dtype(), c2_val / c3);
          return RecursiveRewrite(floordiv(x.Eval() * c1_div + floordiv(y.Eval(), c3), c2_div));
        }
      }
    }
    if ((floordiv(x * broadcast(c1, lanes) + y, broadcast(c2, lanes))).Match(ret)) {
      int64_t c1_val = c1.Eval()->value;
      int64_t c2_val = c2.Eval()->value;
      if (c1_val > 0 && c2_val > 0) {
        int64_t c3 = ZeroAwareGCD(c1_val, c2_val);
        if (c3 > 1) {
          IntImm c1_div = IntImm(c1.Eval().dtype(), c1_val / c3);
          IntImm c2_div = IntImm(c2.Eval().dtype(), c2_val / c3);
          return RecursiveRewrite(floordiv(
              x.Eval() * Broadcast(c1_div, lanes.Eval()) +
                  floordiv(y.Eval(), Broadcast(IntImm(c1.Eval().dtype(), c3), lanes.Eval())),
              Broadcast(c2_div, lanes.Eval())));
        }
      }
    }

    // floordiv(x*c1*c3 + y + z, c2*c3) = floordiv(x*c1 + floordiv(y + z, c3), c2)
    if ((floordiv(x * c1 + y + z, c2)).Match(ret)) {
      int64_t c1_val = c1.Eval()->value;
      int64_t c2_val = c2.Eval()->value;
      if (c1_val > 0 && c2_val > 0) {
        int64_t c3 = ZeroAwareGCD(c1_val, c2_val);
        if (c3 > 1) {
          IntImm c1_div = IntImm(c1.Eval().dtype(), c1_val / c3);
          IntImm c2_div = IntImm(c2.Eval().dtype(), c2_val / c3);
          return RecursiveRewrite(
              floordiv(x.Eval() * c1_div + floordiv(y.Eval() + z.Eval(), c3), c2_div));
        }
      }
    }
    if ((floordiv(x * broadcast(c1, lanes) + y + z, broadcast(c2, lanes))).Match(ret)) {
      int64_t c1_val = c1.Eval()->value;
      int64_t c2_val = c2.Eval()->value;
      if (c1_val > 0 && c2_val > 0) {
        int64_t c3 = ZeroAwareGCD(c1_val, c2_val);
        if (c3 > 1) {
          IntImm c1_div = IntImm(c1.Eval().dtype(), c1_val / c3);
          IntImm c2_div = IntImm(c2.Eval().dtype(), c2_val / c3);
          return RecursiveRewrite(
              floordiv(x.Eval() * Broadcast(c1_div, lanes.Eval()) +
                           floordiv(y.Eval() + z.Eval(),
                                    Broadcast(IntImm(c1.Eval().dtype(), c3), lanes.Eval())),
                       Broadcast(c2_div, lanes.Eval())));
        }
      }
    }

    return ret;
  }

  PrimExpr VisitExpr_(const LENode* op) { return this->VisitExpr(Not(op->b < op->a)); }

  PrimExpr VisitExpr_(const GTNode* op) { return this->VisitExpr(op->b < op->a); }

  PrimExpr VisitExpr_(const GENode* op) { return this->VisitExpr(Not(op->a < op->b)); }

  PrimExpr VisitExpr_(const LTNode* op) {
    PrimExpr a = VisitExpr(op->a);
    PrimExpr b = VisitExpr(op->b);
    PrimExpr ret = tir::LT(a, b);
    // Pattern var to match any expression
    PVar<PrimExpr> x;
    // Pattern var match IntImm
    PVar<IntImm> c1, c2;
    // x < c2 <=> x/c2 < 1 <=> floor(x / c2) < 1
    TRY_RECURSIVE_REWRITE_IF(x<c2, floordiv(x, c2) < 1, c2.Eval()->value> 0);  // NOLINT
    return ret;
  }

  PrimExpr VisitExpr_(const NotNode* op) {
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
    // Pattern var to match any expression
    PVar<PrimExpr> x, y;
    TRY_REWRITE(!(!x), x);
    TRY_REWRITE(!(x <= y), y < x);
    TRY_REWRITE(!(x >= y), x < y);
    TRY_REWRITE(!(x < y), y <= x);
    TRY_REWRITE(!(x > y), x <= y);
    return ret;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  // Recursive rewrite x
  // we limit maximum depth of recursive rewrite allowed to
  // avoid infinite loop
  PrimExpr RecursiveRewrite(const PrimExpr& x) {
    if (recur_depth_ >= kMaxRecurDepth) return x;
    ++recur_depth_;
    PrimExpr res = this->VisitExpr(x);
    --recur_depth_;
    return res;
  }

 private:
  // counter to record recursive rewrite depth.
  int recur_depth_{0};
  // maximum number of recursion allowed during a single pass.
  static const constexpr int kMaxRecurDepth = 5;
};

namespace transform {

Pass RenormalizeSplitPattern() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    arith::Analyzer analyzer;
    n->body = SplitPatternReNormalizer(&analyzer)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RenormalizeSplitPattern", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RenormalizeSplitPattern")
    .set_body_typed(RenormalizeSplitPattern);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
