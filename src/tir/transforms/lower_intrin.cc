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
 *  Lower intrinsic calls and ops to device specific ir when possible.
 * \file lower_intrin.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include <limits>
#include <unordered_set>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

class IntrinInjecter : public tvm::arith::IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;
  using FLowerGeneral = runtime::TypedPackedFunc<PrimExpr(PrimExpr)>;

  IntrinInjecter(arith::Analyzer* analyzer, std::string target, std::string mtriple = "")
      : IRMutatorWithAnalyzer(analyzer) {
    std::vector<std::string> patterns;
    patterns.push_back(target + ".FLowerIntrinsic");
    patterns.push_back(target + ".FLegalize");
    bool is_llvm_aarch64 = (mtriple.find("aarch64") != std::string::npos);
    if (is_llvm_aarch64) {
      patterns.push_back(target + ".aarch64.FLowerIntrinsic");
      patterns.push_back(target + ".aarch64.FLegalize");
    }
    patterns.push_back("default.FLowerIntrinsic");
    patterns.push_back("default.FLegalize");

    for (const std::string& pattern : patterns)
      if (Op::HasAttrMap(pattern)) {
        attr_maps_.push_back(Op::GetAttrMap<FLowerGeneral>(pattern));
        if (fma_ == nullptr) {
          fma_ = (*attr_maps_.rbegin()).get(Op::Get("tir.fma"), nullptr);
        }
      }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (auto* ptr_op = op->op.as<OpNode>()) {
      for (const auto& f_attr_map : attr_maps_) {
        FLowerGeneral f = f_attr_map.get(GetRef<Op>(ptr_op), nullptr);
        if (f != nullptr) {
          PrimExpr e = GetRef<PrimExpr>(op);
          PrimExpr r = f(e);
          ICHECK(r.defined()) << "intrinsic rule must always return valid Expr";
          if (!r.same_as(e)) {
            r = this->VisitExpr(r);
            if (r.defined()) {
              return r;
            }
          }
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const AddNode* op) final {
    if (const MulNode* mb = op->b.as<MulNode>()) {
      return MakeFMA(mb->a, mb->b, op->a, op);
    } else if (const MulNode* ma = op->a.as<MulNode>()) {
      return MakeFMA(ma->a, ma->b, op->b, op);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  // We use floordiv for integer analysis,
  // but will need to lower them to native truncdiv instructions
  PrimExpr VisitExpr_(const FloorDivNode* op) final {
    auto e = GetRef<PrimExpr>(op);
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
    op = ret.as<FloorDivNode>();
    if (op == nullptr) return ret;
    int shift;
    const DataType& dtype = op->dtype;
    ICHECK(dtype.is_int() || dtype.is_uint());

    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      // lower to right shift if possible.
      return op->a >> make_const(dtype, shift);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common path, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0) || analyzer_->CanProveGreaterEqual(e, 0)) {
        return truncdiv(op->a, op->b);
      }

      // If the numerator's lower bound is known, express the floordiv
      // in terms of truncdiv using only positive operands.
      arith::ConstIntBound const_int_bound = analyzer_->const_int_bound(op->a);
      if (const_int_bound->min_value < 0 &&
          const_int_bound->min_value >
              -(Downcast<IntImm>(tvm::max_value(op->a->dtype.element_of()))->value)) {
        // The goal is to write floordiv(a,b) in terms of truncdiv, without using
        // negative operands.
        //
        // For any integer c
        //
        //   floordiv(a,b) == floordiv(a + b*c - b*c, b)
        //                 == floordiv(a + b*c, b) - c
        //
        // Choosing `c = ceildiv(-a_min, b)`.  This can be rewritten in terms of
        // truncdiv as follows.
        //
        //   c == ceildiv(-a_min,b)
        //     == floordiv(-a_min + (b-1), b)
        //     == truncdiv(-a_min + (b-1), b)
        //
        // When substituted into `a + b*c`, this results in a positive argument.
        //
        //   a + b*c
        //     == a + b*ceildiv(-a_min,b)
        //     == a - b*floordiv(a_min,b)
        //     >= a - b*floordiv(a,b)
        //     == floormod(a, b)
        //     >= 0
        //
        // Since the argument is positive, this allows floordiv to be written as
        // followed.
        //
        //   floordiv(a,b)
        //     == floordiv(a + b*c, b) - c
        //     == truncdiv(a + b*c, b) - c
        IntImm min(op->a->dtype.element_of(), const_int_bound->min_value);
        PrimExpr ceildiv = truncdiv((op->b - 1) - min, op->b);
        PrimExpr offset_numerator = analyzer_->Simplify(op->a + op->b * ceildiv);
        return truncdiv(offset_numerator, op->b) - ceildiv;
      }

      DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divident";
      PrimExpr rdiv = truncdiv(op->a, op->b);
      PrimExpr rmod = truncmod(op->a, op->b);
      // condition on b >= 0.
      // truncmod(a, b) < 0 will implies ceildiv,
      // So we need to correct these cases.
      if ((dtype == DataType::Int(32) || dtype == DataType::Int(64)) && support_bitwise_op_) {
        // equivalent to rdiv + (rmod >= 0 ? 0: -1);
        return rdiv + (rmod >> make_const(dtype, dtype.bits() - 1));
      } else {
        return tir::Select(rmod >= 0, rdiv, rdiv - make_const(dtype, 1));
      }

    } else {
      if (dtype.is_float()) {
        // floor(a / b)
        return VisitExpr_(tvm::floor(op->a / op->b).as<CallNode>());
      } else {
        // uncommon case
        DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divisor";
        auto rmod = tir::Var("rmod", dtype);
        auto rdiv = tir::Var("rdiv", dtype);
        // b >= 0 => (rmod >=0 ? rdiv : rdiv - 1)
        // b < 0  => (rmod <= 0 ? rdiv : rdiv - 1)
        PrimExpr let_rdiv =
            tir::Let(rdiv, truncdiv(op->a, op->b),
                     tir::Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0), rdiv,
                                 rdiv - make_const(dtype, 1)));
        return Let(rmod, truncmod(op->a, op->b), let_rdiv);
      }
    }
  }

  PrimExpr VisitExpr_(const FloorModNode* op) final {
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op);
    op = ret.as<FloorModNode>();
    if (op == nullptr) return ret;
    // Lower floordiv to native truncdiv.
    int shift;
    const DataType& dtype = op->dtype;
    ICHECK(dtype.is_int() || dtype.is_uint());

    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      // lower to masking if possible.
      int64_t mask = (static_cast<int64_t>(1) << static_cast<int64_t>(shift)) - 1;
      return op->a & make_const(dtype, mask);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common pass, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0)) {
        return truncmod(op->a, op->b);
      }

      // If the numerator's lower bound is known, express the floormod
      // in terms of truncmod using only positive operands.
      arith::ConstIntBound const_int_bound = analyzer_->const_int_bound(op->a);
      if (const_int_bound->min_value < 0 &&
          const_int_bound->min_value >
              -(Downcast<IntImm>(tvm::max_value(op->a->dtype.element_of()))->value)) {
        // The goal is to write floormod(a,b) in terms of truncdiv and truncmod,
        // without using negative operands.
        //
        // For any integer c
        //
        //   floormod(a, b) == floormod(a + b*c, b)
        //
        // Choosing `c = ceildiv(-a_min, b)`.  This can be rewritten in terms of
        // truncdiv as follows.
        //
        //   c == ceildiv(-a_min,b)
        //     == floordiv(-a_min + (b-1), b)
        //     == truncdiv(-a_min + (b-1), b)
        //
        // When substituted into `a + b*c`, this results in a positive argument.
        //
        //   a + b*c
        //     == a + b*ceildiv(-a_min,b)
        //     == a - b*floordiv(a_min,b)
        //     >= a - b*floordiv(a,b)
        //     == floormod(a, b)
        //     >= 0
        //
        // Since the argument is positive, this allows floordiv to be written as
        // followed.
        //
        //   floormod(a,b)
        //     == floormod(a + b*c, b)
        //     == truncmod(a + b*c, b)
        IntImm min(op->a->dtype.element_of(), const_int_bound->min_value);
        PrimExpr ceildiv = truncdiv(-min + (op->b - 1), op->b);
        PrimExpr offset_numerator = analyzer_->Simplify(op->a + op->b * ceildiv);
        return truncmod(offset_numerator, op->b);
      }

      DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divident";
      // NOTE:condition on b >= 0.
      // mod(a, b) < 0 will imply we are doing ceildiv,
      // So we need to correct these cases.
      PrimExpr rmod = truncmod(op->a, op->b);
      if ((dtype == DataType::Int(32) || dtype == DataType::Int(64)) && support_bitwise_op_) {
        // (rmod >> shift) & b
        // -> (rmod >= 0 ? 0: -1) & b
        // -> rmod >= 0 ? 0 : b
        return rmod + (op->b & (rmod >> make_const(dtype, dtype.bits() - 1)));
      } else {
        return tir::Select(rmod >= 0, rmod, rmod + op->b);
      }

    } else {
      if (dtype.is_float()) {
        // a - floor(a / b) * b
        return op->a - (VisitExpr_(tvm::floor(op->a / op->b).as<CallNode>()) * op->b);
      } else {
        // uncommon case
        DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divsor and divident";
        auto rmod = tir::Var("rmod", dtype);
        // b > 0 && rmod >= 0 -> rmod
        // b > 0 && rmod < 0  -> rmod + b
        // b < 0 && rmod < 0 -> rmod
        // b < 0 && rmod > 0 -> rmod + b
        return Let(
            rmod, truncmod(op->a, op->b),
            Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0), rmod, rmod + op->b));
      }
    }
  }

  PrimExpr VisitExpr_(const MaxNode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    auto e = GetRef<PrimExpr>(op);
    if (max(floordiv(x, y), c).Match(e) && c.Eval()->value >= 0 &&
        analyzer_->CanProveGreaterEqual(y.Eval(), 0)) {
      return max(VisitExpr(truncdiv(x, y).Eval()), c.Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const EQNode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    auto e = GetRef<PrimExpr>(op);
    if ((floormod(x, y) == 0).Match(e)) {
      return VisitExpr((truncmod(x, y) == 0).Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const NENode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    auto e = GetRef<PrimExpr>(op);
    if ((floormod(x, y) != 0).Match(e)) {
      return VisitExpr((truncmod(x, y) != 0).Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

 private:
  PrimExpr SwapBroadcastCast(const PrimExpr& e) {
    // Try to change broadcast(cast(x)) to cast(broadcast(x))
    // For some targets, LLVM will generate more efficient FMA
    // instruction with the latter. For example, vmla vs. vmlal
    // on ARM.
    if (const BroadcastNode* bcast = e.as<BroadcastNode>()) {
      if (const CastNode* cast = bcast->value.as<CastNode>()) {
        auto should_swap = [&]() {
          // Maintain behaviour (int8 -> int16, fp16 -> fp32).
          if (cast->dtype.bits() == cast->value.dtype().bits() * 2) {
            return true;
          }
          // Check both operands are integer-like.
          if (!cast->dtype.is_uint() && !cast->dtype.is_int()) {
            return false;
          }
          if (!cast->value.dtype().is_uint() && !cast->value.dtype().is_int()) {
            return false;
          }
          // If both are integer-like, swap if we have a widening cast.
          return cast->dtype.bits() > cast->value.dtype().bits();
        };

        if (should_swap()) {
          PrimExpr new_bcast = Broadcast(cast->value, bcast->lanes);
          return Cast(bcast->dtype, new_bcast);
        }
      }
    }
    return e;
  }

  PrimExpr MakeFMA(const PrimExpr& a, const PrimExpr& b, const PrimExpr& c, const AddNode* op) {
    // emit fma instruction: a * b + c
    PrimExpr lhs = SwapBroadcastCast(a);
    PrimExpr rhs = SwapBroadcastCast(b);

    if (fma_ != nullptr && op->dtype.is_float()) {
      PrimExpr r = fma_(Call(op->dtype, builtin::fma(), {lhs, rhs, c}));
      if (r.defined()) return this->VisitExpr(r);
    } else {
      if (!lhs.same_as(a) || !rhs.same_as(b)) {
        PrimExpr mul = this->VisitExpr(Mul(lhs, rhs));
        return Add(mul, this->VisitExpr(c));
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  // attribute maps, shared only when FLegalize == FLowerIntrinsic
  std::vector<OpAttrMap<FLowerGeneral>> attr_maps_;
  FLowerGeneral fma_{nullptr};
  bool support_bitwise_op_{true};
};

Stmt LowerIntrinStmt(Stmt stmt, const std::string& target) {
  arith::Analyzer analyzer;
  return IntrinInjecter(&analyzer, target)(std::move(stmt));
}

namespace transform {

Pass LowerIntrin() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "LowerIntrin: Require the target attribute";
    arith::Analyzer analyzer;
    auto mtriple = target.value()->GetAttr<runtime::String>("mtriple", "");
    n->body =
        IntrinInjecter(&analyzer, target.value()->kind->name, mtriple.value())(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerIntrin", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerIntrin").set_body_typed(LowerIntrin);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
