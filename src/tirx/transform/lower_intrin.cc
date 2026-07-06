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
#include <tvm/ir/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_set>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tirx {

static Expr LowerAccessPtr(const CallNode* call) {
  TVM_FFI_ICHECK_EQ(call->args.size(), 5U);
  PrimType dtype = call->args[0].as_or_throw<PrimExpr>().ty();
  PrimExpr offset = call->args[2].as_or_throw<PrimExpr>();
  TVM_FFI_ICHECK(call->ty.as<PointerTypeNode>());

  // An access pointer may itself be used as the base of another access
  // pointer.  Fold those offsets before constructing the synthetic
  // BufferLoad so lowering never assumes that args[1] is immediately a Var.
  Expr buffer = call->args[1];
  while (const auto* inner = buffer.as<CallNode>()) {
    if (!inner->op.same_as(builtin::tvm_access_ptr())) break;
    TVM_FFI_ICHECK_EQ(inner->args.size(), 5U);
    PrimType inner_dtype = inner->args[0].as_or_throw<PrimExpr>().ty();
    TVM_FFI_ICHECK_EQ(inner_dtype, dtype)
        << "Nested tvm_access_ptr calls must use the same element type";
    PrimExpr inner_offset = inner->args[2].as_or_throw<PrimExpr>();
    if (inner_offset.ty() != offset.ty()) {
      inner_offset = Cast(offset.ty(), inner_offset);
    }
    offset = inner_offset + offset;
    buffer = inner->args[1];
  }

  const auto* buffer_node = buffer.as<VarNode>();
  TVM_FFI_ICHECK(buffer_node)
      << "tvm_access_ptr expects a buffer Var or nested tvm_access_ptr as args[1], but got "
      << buffer;
  Var buffer_var = ffi::GetRef<Var>(buffer_node);
  if (dtype.lanes() != 1) {
    PrimType offset_ty = offset.ty();
    offset = offset * IntImm(offset_ty, dtype.lanes());
    offset = Ramp(offset, IntImm(offset_ty, 1), dtype.lanes());
  }
  Buffer dummy_buf(buffer_var, dtype.WithLanes(1), {offset + 1}, {}, 0, buffer_var->name_hint, 0, 0,
                   kDefault);
  BufferLoad buf_load(dummy_buf, {offset});
  return Call(call->ty, builtin::address_of(), {buf_load});
}

class IntrinInjecter : public tvm::arith::IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::VisitExpr_;
  using IRMutatorWithAnalyzer::VisitStmt_;
  using FLowerGeneral = ffi::TypedFunction<PrimExpr(PrimExpr)>;

  IntrinInjecter(const arith::Analyzer& analyzer, const Target& tgt, bool enable_fast_math)
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
          static const Op& fma_op = Op::Get("tirx.fma");
          fma_ = (*attr_maps_.rbegin()).get(fma_op, nullptr);
        }
      }
  }

  Expr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      return this->VisitExpr(LowerAccessPtr(op));
    }
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
              r = this->VisitPrimExpr(r);
              if (r.defined()) {
                return r;
              }
            }
          }
        }
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Expr VisitExpr_(const AddNode* op) final {
    if (const MulNode* mb = op->b.as<MulNode>()) {
      return MakeFMA(mb->a, mb->b, op->a, op);
    } else if (const MulNode* ma = op->a.as<MulNode>()) {
      return MakeFMA(ma->a, ma->b, op->b, op);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  // We use floordiv for integer analysis,
  // but will need to lower them to native truncdiv instructions
  Expr VisitExpr_(const FloorDivNode* op) final {
    auto e = ffi::GetRef<PrimExpr>(op);
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op).as_or_throw<PrimExpr>();
    op = ret.as<FloorDivNode>();
    if (op == nullptr) return ret;
    int shift;
    PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt));

    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      // lower to right shift if possible.
      return op->a >> IntImm(dtype, shift);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common path, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0) || analyzer_->CanProveGreaterEqual(e, 0)) {
        return truncdiv(op->a, op->b);
      }
      if (const IntImmNode* b_as_intimm = op->b.as<IntImmNode>()) {
        int64_t b_value = b_as_intimm->value;
        if (auto opt_c_value = TryFindShiftCoefficientForPositiveRange(op->a, b_value)) {
          int64_t c_value = *opt_c_value;
          // now we can safely lower to truncdiv
          return truncdiv(op->a + IntImm(dtype, b_value * c_value), op->b) - IntImm(dtype, c_value);
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
        return tirx::Select(rmod >= 0, rdiv, rdiv - MakeConst(dtype, 1));
      }

    } else {
      if (dtype.code() == DLDataTypeCode::kDLFloat) {
        // floor(a / b)
        return VisitExpr_(tvm::floor(op->a / op->b).as<CallNode>());
      } else {
        // uncommon case
        DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divisor";
        PrimVar rmod("rmod", dtype);
        PrimVar rdiv("rdiv", dtype);
        // b >= 0 => (rmod >=0 ? rdiv : rdiv - 1)
        // b < 0  => (rmod <= 0 ? rdiv : rdiv - 1)
        PrimExpr let_rdiv =
            tirx::Let(rdiv, truncdiv(op->a, op->b),
                      tirx::Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0), rdiv,
                                   rdiv - MakeConst(dtype, 1)));
        return Let(rmod, truncmod(op->a, op->b), let_rdiv);
      }
    }
  }

  Expr VisitExpr_(const FloorModNode* op) final {
    PrimExpr ret = IRMutatorWithAnalyzer::VisitExpr_(op).as_or_throw<PrimExpr>();
    op = ret.as<FloorModNode>();
    if (op == nullptr) return ret;
    // Lower floordiv to native truncdiv.
    int shift;
    PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt));

    if (support_bitwise_op_ && is_const_power_of_two_integer(op->b, &shift)) {
      // lower to masking if possible.
      int64_t mask = (static_cast<int64_t>(1) << static_cast<int64_t>(shift)) - 1;
      return op->a & IntImm(dtype, mask);
    }

    if (analyzer_->CanProveGreaterEqual(op->b, 0)) {
      // Common pass, positive divisor
      if (analyzer_->CanProveGreaterEqual(op->a, 0)) {
        return truncmod(op->a, op->b);
      }
      if (const IntImmNode* b_as_intimm = op->b.as<IntImmNode>()) {
        int64_t b_value = b_as_intimm->value;
        if (auto opt_c_value = TryFindShiftCoefficientForPositiveRange(op->a, b_value)) {
          int64_t c_value = *opt_c_value;
          // floormod(a, b) == floormod(a + b*c, b)  == truncmod(a + b*c, b)
          return truncmod(op->a + IntImm(dtype, c_value * b_value), op->b);
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
        return tirx::Select(rmod >= 0, rmod, rmod + op->b);
      }

    } else {
      if (dtype.code() == DLDataTypeCode::kDLFloat) {
        // a - floor(a / b) * b
        return op->a -
               (VisitExpr_(tvm::floor(op->a / op->b).as<CallNode>()).as_or_throw<PrimExpr>() *
                op->b);
      } else {
        // uncommon case
        DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divsor and divident";
        PrimVar rmod("rmod", dtype);
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

  Expr VisitExpr_(const MaxNode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    auto e = ffi::GetRef<PrimExpr>(op);
    if (max(floordiv(x, y), c).Match(e) && c.Eval()->value >= 0 &&
        analyzer_->CanProveGreaterEqual(y.Eval(), 0)) {
      return max(VisitPrimExpr(truncdiv(x, y).Eval()), c.Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Expr VisitExpr_(const EQNode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    auto e = ffi::GetRef<PrimExpr>(op);
    if ((floormod(x, y) == 0).Match(e)) {
      return VisitPrimExpr((truncmod(x, y) == 0).Eval());
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Expr VisitExpr_(const NENode* op) final {
    using namespace arith;
    PVar<PrimExpr> x, y;
    auto e = ffi::GetRef<PrimExpr>(op);
    if ((floormod(x, y) != 0).Match(e)) {
      return VisitPrimExpr((truncmod(x, y) != 0).Eval());
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
          PrimExpr new_bcast = Broadcast(cast->value, bcast->lanes);
          return Cast(bcast->ty.as_or_throw<PrimType>(), new_bcast);
        }
      }
    }
    return e;
  }

  PrimExpr MakeFMA(const PrimExpr& a, const PrimExpr& b, const PrimExpr& c, const AddNode* op) {
    // emit fma instruction: a * b + c
    PrimExpr lhs = SwapBroadcastCast(a);
    PrimExpr rhs = SwapBroadcastCast(b);

    if (fma_ != nullptr && op->ty.as_or_throw<PrimType>().code() == DLDataTypeCode::kDLFloat) {
      PrimExpr r = fma_(Call(op->ty.as_or_throw<PrimType>(), builtin::fma(), {lhs, rhs, c})
                            .as_or_throw<PrimExpr>());
      if (r.defined()) return this->VisitPrimExpr(r);
    } else {
      if (!lhs.same_as(a) || !rhs.same_as(b)) {
        PrimExpr mul = this->VisitPrimExpr(Mul(lhs, rhs));
        return Add(mul, this->VisitPrimExpr(c));
      }
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op).as_or_throw<PrimExpr>();
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
    arith::ConstIntBound const_int_bound_a = analyzer_->const_int_bound(a);
    if (const_int_bound_a->min_value >= 0) {
      return std::nullopt;
    }
    PrimType a_ty = a.ty();
    // This overflow check is scalar element based. Lane count is intentionally ignored.
    const int64_t max_value_of_dtype =
        tvm::max_value(PrimType(a_ty.code(), a_ty.bits())).as_or_throw<IntImm>()->value;

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
  arith::Analyzer analyzer;
  bool enable_fast_math =
      transform::PassContext::Current()->GetConfig<bool>("tirx.enable_fast_math", false).value();
  return IntrinInjecter(analyzer, Target(ffi::String(target)), enable_fast_math)(std::move(stmt));
}

namespace transform {

Pass LowerIntrin() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_ICHECK(target.has_value()) << "LowerIntrin: Require the target attribute";
    arith::Analyzer analyzer;
    bool enable_fast_math = ctx->GetConfig<bool>("tirx.enable_fast_math", false).value();
    n->body = IntrinInjecter(analyzer, target.value(), enable_fast_math)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerIntrin", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerIntrin", LowerIntrin);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
