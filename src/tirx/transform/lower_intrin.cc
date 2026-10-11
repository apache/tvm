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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_set>

#include "../../sym/pattern_match.h"
#include "../ir/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

class IntrinInjecter : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::Mutate;
  using IRMutatorWithAnalyzer::Mutate_;

  using FLowerGeneral = ffi::TypedFunction<PrimExpr(PrimExpr)>;

  IntrinInjecter(const sym::Analyzer& analyzer, const Target& tgt, bool enable_fast_math)
      : IRMutatorWithAnalyzer(analyzer) {
    std::string target = tgt->kind->name;
    ffi::String mtriple = tgt->GetAttr<ffi::String>("mtriple").value_or("");

    std::vector<std::string> patterns;
    // Add the fast math patterns when requested.  The priority of the fast math
    // patterns is higher than the normal patterns.
    if (enable_fast_math) {
      patterns.push_back(target + ".fastmath.FLowerIntrinsic");
      patterns.push_back(target + ".fastmath.FLegalize");
    }
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
          static const Op fma_op = Op::Get("prim.fma");
          fma_ = (*attr_maps_.rbegin()).get(fma_op, nullptr);
        }
      }
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (auto* ptr_op = op->op.as<OpNode>()) {
      Op op_ref = ffi::GetRef<Op>(ptr_op);
      Expr e = ffi::GetRef<Call>(op);
      if (auto prim_e = e.as<PrimExpr>()) {
        for (const auto& f_attr_map : attr_maps_) {
          FLowerGeneral f = f_attr_map.get(op_ref, nullptr);
          if (f != nullptr) {
            PrimExpr r = f(prim_e.value());
            TVM_FFI_ICHECK(r.defined()) << "intrinsic rule must always return valid Expr";
            if (!r.same_as(prim_e.value())) {
              r = this->Mutate(r, inplace_mode).ValueOrUnchanged(r);
              if (r.defined()) {
                return r;
              }
            }
          }
        }
      }
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* op, InplaceMode inplace_mode) final {
    if (const prim::MulNode* mb = op->b.as<prim::MulNode>()) {
      return MakeFMA(mb->a, mb->b, op->a, op);
    } else if (const prim::MulNode* ma = op->a.as<prim::MulNode>()) {
      return MakeFMA(ma->a, ma->b, op->b, op);
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  // We use floordiv for integer analysis,
  // but will need to lower them to native truncdiv instructions
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode* op, InplaceMode inplace_mode) final {
    auto e = ffi::GetRef<PrimExpr>(op);
    PrimExpr ret = IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    op = ret.as<prim::FloorDivNode>();
    if (op == nullptr) return ret;
    int shift;
    PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt));

    if (support_bitwise_op_ && IsPowerOfTwoInt(op->b, &shift)) {
      // lower to right shift if possible.
      return op->a >> IntImm(dtype, shift);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common path, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0) || analyzer_->CanProveGreaterEqual(e, 0)) {
        return truncdiv(op->a, op->b);
      }
      const auto* b_as_intimm = op->b.as<IntImmNode>();
      if (auto b_value = b_as_intimm ? b_as_intimm->value.as<int64_t>() : std::nullopt;
          b_value.has_value()) {
        if (auto opt_c_value = TryFindShiftCoefficientForPositiveRange(op->a, *b_value);
            opt_c_value.has_value()) {
          int64_t c_value = *opt_c_value;
          // now we can safely lower to truncdiv
          return truncdiv(op->a + IntImm(dtype, b_as_intimm->value * c_value), op->b) -
                 IntImm(dtype, c_value);
        }
      }
      DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divident";
      PrimExpr rdiv = truncdiv(op->a, op->b);
      PrimExpr rmod = truncmod(op->a, op->b);
      // condition on b >= 0.
      // truncmod(a, b) < 0 will implies ceildiv,
      // So we need to correct these cases.
      if ((dtype == PrimType::Int(32) || dtype == PrimType::Int(64)) && support_bitwise_op_) {
        // equivalent to rdiv + (rmod >= 0 ? 0: -1);
        return rdiv + (rmod >> IntImm(dtype, dtype.bits() - 1));
      } else {
        return prim::Select(rmod >= 0, rdiv, rdiv - MakeConst(dtype, 1));
      }

    } else {
      if (dtype.code() == DLDataTypeCode::kDLFloat) {
        // floor(a / b)
        PrimExpr lowered = tvm::prim::floor(op->a / op->b);
        return Mutate(lowered, inplace_mode).ValueOrUnchanged(lowered);
      } else {
        // uncommon case
        DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divisor";
        PrimVar rmod("rmod", dtype);
        PrimVar rdiv("rdiv", dtype);
        // b >= 0 => (rmod >=0 ? rdiv : rdiv - 1)
        // b < 0  => (rmod <= 0 ? rdiv : rdiv - 1)
        PrimExpr let_rdiv =
            prim::Let(rdiv, truncdiv(op->a, op->b),
                      prim::Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0), rdiv,
                                   rdiv - MakeConst(dtype, 1)));
        return prim::Let(rmod, truncmod(op->a, op->b), let_rdiv);
      }
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode* op, InplaceMode inplace_mode) final {
    PrimExpr ret = IRMutatorWithAnalyzer::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    op = ret.as<prim::FloorModNode>();
    if (op == nullptr) return ret;
    // Lower floordiv to native truncdiv.
    int shift;
    PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt));

    if (support_bitwise_op_ && IsPowerOfTwoInt(op->b, &shift)) {
      // lower to masking if possible.
      ffi::BigInt mask = (ffi::BigInt(1) << shift) - 1;
      return op->a & IntImm(dtype, mask);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common pass, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0)) {
        return truncmod(op->a, op->b);
      }
      const auto* b_as_intimm = op->b.as<IntImmNode>();
      if (auto b_value = b_as_intimm ? b_as_intimm->value.as<int64_t>() : std::nullopt;
          b_value.has_value()) {
        if (auto opt_c_value = TryFindShiftCoefficientForPositiveRange(op->a, *b_value);
            opt_c_value.has_value()) {
          int64_t c_value = *opt_c_value;
          // floormod(a, b) == floormod(a + b*c, b)  == truncmod(a + b*c, b)
          return truncmod(op->a + IntImm(dtype, b_as_intimm->value * c_value), op->b);
        }
      }
      DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divident";
      // NOTE:condition on b >= 0.
      // mod(a, b) < 0 will imply we are doing ceildiv,
      // So we need to correct these cases.
      PrimExpr rmod = truncmod(op->a, op->b);
      if ((dtype == PrimType::Int(32) || dtype == PrimType::Int(64)) && support_bitwise_op_) {
        // (rmod >> shift) & b
        // -> (rmod >= 0 ? 0: -1) & b
        // -> rmod >= 0 ? 0 : b
        return rmod + (op->b & (rmod >> IntImm(dtype, dtype.bits() - 1)));
      } else {
        return prim::Select(rmod >= 0, rmod, rmod + op->b);
      }

    } else {
      if (dtype.code() == DLDataTypeCode::kDLFloat) {
        // a - floor(a / b) * b
        PrimExpr lowered = tvm::prim::floor(op->a / op->b);
        return op->a - (Mutate(lowered, inplace_mode).ValueOrUnchanged(lowered) * op->b);
      } else {
        // uncommon case
        DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divsor and divident";
        PrimVar rmod("rmod", dtype);
        // b > 0 && rmod >= 0 -> rmod
        // b > 0 && rmod < 0  -> rmod + b
        // b < 0 && rmod < 0 -> rmod
        // b < 0 && rmod > 0 -> rmod + b
        return prim::Let(rmod, truncmod(op->a, op->b),
                         prim::Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0), rmod,
                                      rmod + op->b));
      }
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode* op, InplaceMode inplace_mode) final {
    using namespace sym;
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    auto e = ffi::GetRef<PrimExpr>(op);
    if (max(floordiv(x, y), c).Match(e) && c.Eval()->value >= 0 &&
        analyzer_->CanProveGreaterEqual(y.Eval(), 0)) {
      PrimExpr input = truncdiv(x, y).Eval();
      return max(Mutate(input, inplace_mode).ValueOrUnchanged(input), c.Eval());
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* op, InplaceMode inplace_mode) final {
    using namespace sym;
    PVar<PrimExpr> x, y;
    auto e = ffi::GetRef<PrimExpr>(op);
    if ((floormod(x, y) == 0).Match(e)) {
      PrimExpr input = (truncmod(x, y) == 0).Eval();
      return Mutate(input, inplace_mode).ValueOrUnchanged(input);
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NENode* op, InplaceMode inplace_mode) final {
    using namespace sym;
    PVar<PrimExpr> x, y;
    auto e = ffi::GetRef<PrimExpr>(op);
    if ((floormod(x, y) != 0).Match(e)) {
      PrimExpr input = (truncmod(x, y) != 0).Eval();
      return Mutate(input, inplace_mode).ValueOrUnchanged(input);
    }
    return IRMutatorWithAnalyzer::Mutate_(op, inplace_mode);
  }

 private:
  PrimExpr SwapBroadcastCast(const PrimExpr& e) {
    // Try to change broadcast(prim::cast(x)) to prim::cast(broadcast(x))
    // For some targets, LLVM will generate more efficient FMA
    // instruction with the latter. For example, vmla vs. vmlal
    // on ARM.
    if (const prim::BroadcastNode* bcast = e.as<prim::BroadcastNode>()) {
      if (const prim::CastNode* cast = bcast->value.as<prim::CastNode>()) {
        auto should_swap = [&]() {
          PrimType cast_ty = cast->ty.as_or_throw<PrimType>();
          PrimType value_ty = cast->value.ty();
          // Maintain behaviour (int8 -> int16, fp16 -> fp32).
          if (cast_ty.bits() == value_ty.bits() * 2) {
            return true;
          }
          // Check both operands are integer-like.
          if (cast_ty.code() != DLDataTypeCode::kDLUInt &&
              cast_ty.code() != DLDataTypeCode::kDLInt) {
            return false;
          }
          if (value_ty.code() != DLDataTypeCode::kDLUInt &&
              value_ty.code() != DLDataTypeCode::kDLInt) {
            return false;
          }
          // If both are integer-like, swap if we have a widening cast.
          return cast_ty.bits() > value_ty.bits();
        };

        if (should_swap()) {
          PrimExpr new_bcast = prim::Broadcast(cast->value, bcast->lanes);
          return prim::Cast(bcast->ty.as_or_throw<PrimType>(), new_bcast);
        }
      }
    }
    return e;
  }

  PrimExpr MakeFMA(const PrimExpr& a, const PrimExpr& b, const PrimExpr& c,
                   const prim::AddNode* op) {
    // emit fma instruction: a * b + c
    PrimExpr lhs = SwapBroadcastCast(a);
    PrimExpr rhs = SwapBroadcastCast(b);

    if (fma_ != nullptr && op->ty.as_or_throw<PrimType>().code() == DLDataTypeCode::kDLFloat) {
      PrimExpr r = fma_(
          Call(op->ty.as_or_throw<PrimType>(), fma_op(), {lhs, rhs, c}).as_or_throw<PrimExpr>());
      if (r.defined()) return this->Mutate(r, InplaceMode::kDisallow).ValueOrUnchanged(r);
    } else {
      if (!lhs.same_as(a) || !rhs.same_as(b)) {
        PrimExpr input = prim::Mul(lhs, rhs);
        PrimExpr mul = this->Mutate(input, InplaceMode::kDisallow).ValueOrUnchanged(input);
        return prim::Add(mul, this->Mutate(c, InplaceMode::kDisallow).ValueOrUnchanged(c));
      }
    }
    return IRMutatorWithAnalyzer::Mutate_(op, InplaceMode::kDisallow)
        .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
  }

  /*!
   * \brief Try to find a shift co-efficient c such that a + b*c positive and does not overflow.
   *
   * \param a the dividend
   * \param b_value the divisor
   * \return the shift co-efficient c, or nullopt if not found
   */
  std::optional<int64_t> TryFindShiftCoefficientForPositiveRange(const PrimExpr& a,
                                                                 int64_t b_value) {
    if (b_value <= 0) {
      return std::nullopt;
    }
    // NOTE: we need to be very careful in the checks below, to make sure
    // all the intermediate calculations in both compiler checks and runtime checks
    // do not overflow
    sym::ConstIntBound const_int_bound_a = analyzer_->const_int_bound(a);
    if (const_int_bound_a->min_value >= 0) {
      return std::nullopt;
    }
    PrimType a_ty = a.ty();
    // This overflow check is scalar element based. Lane count is intentionally ignored.
    auto dtype_max = tvm::prim::max_value(PrimType(a_ty.code(), a_ty.bits()))
                         .as_or_throw<IntImm>()
                         ->value.as<int64_t>();
    if (!dtype_max.has_value()) return std::nullopt;
    const int64_t max_value_of_dtype = *dtype_max;

    // NOTE: ensures that (b-1) - a_min does not overflow
    // also note: max_value_of_dtype + const_int_bound_a->min_value won't overflow
    // since a_min is negative, adding it to a positive value will not overflow
    if (b_value - 1 > max_value_of_dtype + const_int_bound_a->min_value) {
      return std::nullopt;
    }
    int64_t c_value = ((b_value - 1) - const_int_bound_a->min_value) / b_value;
    TVM_FFI_ICHECK_GT(c_value, 0);
    // NOTE: the c_value * b_value risks in overflow
    if (c_value > max_value_of_dtype / b_value) return std::nullopt;
    // need to check if the offset numerator will overflow
    // to ensure if don't overflow, we need to use max_value_of_dtype - b_value * c_value
    // note that b_value * c_value is positive, max_value_of_dtype is also positive, so the
    // subtraction will not overflow
    if (const_int_bound_a->max_value > max_value_of_dtype - b_value * c_value) {
      // a + b * c risks overflow
      return std::nullopt;
    }
    return c_value;
  }

  std::vector<OpAttrMap<FLowerGeneral>> attr_maps_;
  FLowerGeneral fma_{nullptr};
  bool support_bitwise_op_{true};
};

Stmt LowerIntrinStmt(Stmt stmt, const std::string& target) {
  sym::Analyzer analyzer;
  bool enable_fast_math =
      transform::PassContext::Current()->GetConfig<bool>("tirx.enable_fast_math").value_or(false);
  return ffi::make_object<IntrinInjecter>(analyzer, Target(ffi::String(target)), enable_fast_math)
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(stmt);
}

namespace transform {

Pass LowerIntrin() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_ICHECK(target.has_value()) << "LowerIntrin: Require the target attribute";
    sym::Analyzer analyzer;
    bool enable_fast_math = ctx->GetConfig<bool>("tirx.enable_fast_math").value_or(false);
    n->body = ffi::make_object<IntrinInjecter>(analyzer, target.value(), enable_fast_math)
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(n->body);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.LowerIntrin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerIntrin", LowerIntrin);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
