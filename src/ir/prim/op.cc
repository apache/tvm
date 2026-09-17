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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/op.h>

#include <cmath>

#include "const_fold.h"
#include "op_utils.h"

namespace tvm {
using prim::is_const_int;
using prim::MakeConst;
using namespace prim::detail;
namespace {
// File-local helper: true if `expr` is a call to prim::builtin::vscale().
bool IsVScaleCall(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallNode>()) {
    return call->op.same_as(prim::builtin::vscale());
  }
  return false;
}

}  // namespace
namespace prim::detail {
void BroadcastToMatchLanes(PrimExpr& op_a, PrimExpr& op_b) {  // NOLINT(*)
  PrimType ty_a = op_a.ty();
  PrimType ty_b = op_b.ty();

  if (!ty_a.IsScalableVector() && !ty_a.IsFixedLengthVector() &&
      (ty_b.IsScalableVector() || ty_b.IsFixedLengthVector())) {
    if (ty_b.IsScalableVector()) {
      PrimType i32_ty = PrimType::Int(32);
      op_a = prim::Broadcast(
          op_a, prim::Mul(ty_b.VScaleFactor(),
                          Call(i32_ty, prim::builtin::vscale(), {}).as_or_throw<PrimExpr>()));
    } else {
      op_a = prim::Broadcast(op_a, ty_b.lanes());
    }
  }
}

PrimType PromoteBinaryOpType(PrimType lhs_ty, PrimType rhs_ty) {
  if (lhs_ty->dtype == rhs_ty->dtype) {
    return lhs_ty;
  }

  // Keep conversion behavior consistent with the previous DataType-based path.
  if (IsFloatType(lhs_ty) && IsFloatType(rhs_ty)) {
    return lhs_ty.bits() < rhs_ty.bits() ? rhs_ty : lhs_ty;
  } else if (!IsFloatType(lhs_ty) && IsFloatType(rhs_ty)) {
    return rhs_ty;
  } else if (IsFloatType(lhs_ty) && !IsFloatType(rhs_ty)) {
    return lhs_ty;
  } else if (!IsBFloat16Type(lhs_ty) && IsBFloat16Type(rhs_ty)) {
    return rhs_ty;
  } else if (IsBFloat16Type(lhs_ty) && !IsBFloat16Type(rhs_ty)) {
    return lhs_ty;
  } else if (!IsFloat8Type(lhs_ty) && IsFloat8Type(rhs_ty)) {
    return rhs_ty;
  } else if (IsFloat8Type(lhs_ty) && !IsFloat8Type(rhs_ty)) {
    return lhs_ty;
  } else if (!IsFloat6Type(lhs_ty) && IsFloat6Type(rhs_ty)) {
    return rhs_ty;
  } else if (IsFloat6Type(lhs_ty) && !IsFloat6Type(rhs_ty)) {
    return lhs_ty;
  } else if (!IsFloat4Type(lhs_ty) && IsFloat4Type(rhs_ty)) {
    return rhs_ty;
  } else if (IsFloat4Type(lhs_ty) && !IsFloat4Type(rhs_ty)) {
    return lhs_ty;
  } else if (lhs_ty.MatchesCode(DLDataTypeCode::kDLBool) &&
             rhs_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return rhs_ty;
  } else if (lhs_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
             rhs_ty.MatchesCode(DLDataTypeCode::kDLBool)) {
    return lhs_ty;
  } else if ((lhs_ty.MatchesCode(DLDataTypeCode::kDLInt) &&
              rhs_ty.MatchesCode(DLDataTypeCode::kDLInt)) ||
             (lhs_ty.MatchesCode(DLDataTypeCode::kDLUInt) &&
              rhs_ty.MatchesCode(DLDataTypeCode::kDLUInt))) {
    return lhs_ty.bits() < rhs_ty.bits() ? rhs_ty : lhs_ty;
  } else if ((lhs_ty.MatchesCode(DLDataTypeCode::kDLInt) &&
              rhs_ty.MatchesCode(DLDataTypeCode::kDLUInt)) ||
             (lhs_ty.MatchesCode(DLDataTypeCode::kDLUInt) &&
              rhs_ty.MatchesCode(DLDataTypeCode::kDLInt))) {
    if (lhs_ty.bits() < rhs_ty.bits()) {
      return rhs_ty;
    } else if (lhs_ty.bits() > rhs_ty.bits()) {
      return lhs_ty;
    } else {
      return lhs_ty.MatchesCode(DLDataTypeCode::kDLUInt) ? lhs_ty
                                                         : lhs_ty.WithCode(DLDataTypeCode::kDLUInt);
    }
  } else {
    TVM_FFI_THROW(InternalError) << "Cannot match type " << lhs_ty->dtype << " vs "
                                 << rhs_ty->dtype;
  }
  return lhs_ty;
}

// The public function with a quick checking path.
void BinaryOpMatchTypes(PrimExpr& lhs, PrimExpr& rhs, Span span) {  // NOLINT(*)
  TVM_FFI_CHECK(lhs.defined(), ValueError) << "`lhs` is null in the binary operator";
  TVM_FFI_CHECK(rhs.defined(), ValueError) << "`rhs` is null in the binary operator";
  const PrimTypeNode* lhs_ty_node = GetPrimTypeNode(lhs);
  const PrimTypeNode* rhs_ty_node = GetPrimTypeNode(rhs);
  if (lhs_ty_node == rhs_ty_node || lhs_ty_node->dtype == rhs_ty_node->dtype) return;

  BroadcastToMatchLanes(lhs, rhs);
  BroadcastToMatchLanes(rhs, lhs);

  PrimType lhs_ty = lhs.ty();
  PrimType rhs_ty = rhs.ty();

  TVM_FFI_ICHECK(lhs_ty.IsScalableVector() == rhs_ty.IsScalableVector())
      << "Can't match scalable and fixed length vectors";

  bool lanes_match = false;

  if (lhs_ty.IsScalableVector()) {
    lanes_match = lhs_ty.VScaleFactor() == rhs_ty.VScaleFactor();
  } else {
    lanes_match = lhs_ty.lanes() == rhs_ty.lanes();
  }

  TVM_FFI_ICHECK(lanes_match) << "Cannot match type " << lhs_ty->dtype << " vs " << rhs_ty->dtype;

  PrimType promoted_ty = PromoteBinaryOpType(lhs_ty, rhs_ty);
  if (lhs_ty->dtype != promoted_ty->dtype) {
    lhs = prim::cast(promoted_ty, lhs, span);
  }
  if (rhs_ty->dtype != promoted_ty->dtype) {
    rhs = prim::cast(promoted_ty, rhs, span);
  }
}

}  // namespace prim::detail
namespace prim {
// maximum and min limits
PrimExpr max_value(PrimType value_ty, Span span) {
  PrimType dtype = value_ty;
  TVM_FFI_ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.MatchesCode(DLDataTypeCode::kDLInt)) {
    if (dtype.bits() == 64) {
      return IntImm(value_ty, std::numeric_limits<int64_t>::max(), span);
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = (val << (dtype.bits() - 1)) - 1;
      return IntImm(value_ty, val, span);
    }
  } else if (dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
    if (dtype.bits() == 64) {
      return MakeConst(dtype, std::numeric_limits<uint64_t>::max(), span);
    } else if (dtype.bits() < 64) {
      uint64_t val = 1;
      val = (val << static_cast<uint64_t>(dtype.bits())) - 1;
      return IntImm(value_ty, static_cast<int64_t>(val), span);
    }
  } else if (IsFloatType(dtype)) {
    if (dtype.bits() == 64) {
      return FloatImm(value_ty, std::numeric_limits<double>::max(), span);
    } else if (dtype.bits() == 32) {
      return FloatImm(value_ty, std::numeric_limits<float>::max(), span);
    } else if (dtype.bits() == 16) {
      return FloatImm(value_ty, 65504.0, span);
    }
  } else if (IsBFloat16Type(dtype)) {
    return FloatImm(value_ty, std::numeric_limits<float>::max(), span);
  } else if (IsFloat8Type(dtype)) {
    // according to https://arxiv.org/pdf/2209.05433.pdf
    if (dtype.code() == DLDataTypeCode::kDLFloat8_e5m2) {
      return FloatImm(value_ty, 57344.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e5m2fnuz) {
      return FloatImm(value_ty, 57344.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e4m3fn) {
      return FloatImm(value_ty, 448.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e4m3fnuz ||
               dtype.code() == DLDataTypeCode::kDLFloat8_e4m3) {
      return FloatImm(value_ty, 448.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e4m3b11fnuz) {
      return FloatImm(value_ty, 30.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e3m4) {
      return FloatImm(value_ty, 31.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e8m0fnu) {
      return FloatImm(value_ty, 3.4028236692093846e+38, span);
    }
  } else if (IsFloat6Type(dtype)) {
    if (dtype.code() == DLDataTypeCode::kDLFloat6_e2m3fn) {
      return FloatImm(value_ty, 7.5, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat6_e3m2fn) {
      return FloatImm(value_ty, 28.0, span);
    }
  } else if (IsFloat4Type(dtype)) {
    return FloatImm(value_ty, 6.0, span);
  }
  TVM_FFI_THROW(InternalError) << "Cannot decide max_value for type" << dtype;
}

PrimExpr min_value(PrimType value_ty, Span span) {
  PrimType dtype = value_ty;
  TVM_FFI_ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.MatchesCode(DLDataTypeCode::kDLInt)) {
    if (dtype.bits() == 64) {
      return IntImm(value_ty, std::numeric_limits<int64_t>::lowest(), span);
    } else if (dtype.bits() < 64) {
      int64_t val = 1;
      val = -(val << (dtype.bits() - 1));
      return IntImm(value_ty, val, span);
    }
  } else if (dtype.MatchesCode(DLDataTypeCode::kDLUInt)) {
    return IntImm(value_ty, 0, span);
  } else if (IsFloatType(dtype)) {
    if (dtype.bits() == 64) {
      return FloatImm(value_ty, std::numeric_limits<double>::lowest(), span);
    } else if (dtype.bits() == 32) {
      return FloatImm(value_ty, std::numeric_limits<float>::lowest(), span);
    } else if (dtype.bits() == 16) {
      return FloatImm(value_ty, -65504.0, span);
    }
  } else if (IsBFloat16Type(dtype)) {
    return FloatImm(value_ty, std::numeric_limits<float>::lowest(), span);
  } else if (IsFloat8Type(dtype)) {
    // according to https://arxiv.org/pdf/2209.05433.pdf
    if (dtype.code() == DLDataTypeCode::kDLFloat8_e5m2) {
      return FloatImm(value_ty, -57344.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e5m2fnuz) {
      return FloatImm(value_ty, 0.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e4m3fn) {
      return FloatImm(value_ty, -448.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e4m3fnuz) {
      return FloatImm(value_ty, 0.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e4m3) {
      return FloatImm(value_ty, -448.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e4m3b11fnuz) {
      return FloatImm(value_ty, 0.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e3m4) {
      return FloatImm(value_ty, -31.0, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat8_e8m0fnu) {
      return FloatImm(value_ty, 0.0, span);
    }
  } else if (IsFloat6Type(dtype)) {
    if (dtype.code() == DLDataTypeCode::kDLFloat6_e2m3fn) {
      return FloatImm(value_ty, -7.5, span);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat6_e3m2fn) {
      return FloatImm(value_ty, -28.0, span);
    }
  } else if (IsFloat4Type(dtype)) {
    return FloatImm(value_ty, -6.0, span);
  }
  TVM_FFI_THROW(InternalError) << "Cannot decide min_value for type" << dtype;
}

PrimExpr cast(PrimType t, PrimExpr value, Span span) {
  PrimType dtype = t;
  if (value.ty() == dtype) return value;
  TVM_FFI_CHECK(!value.ty().IsVoid(), TypeError)
      << "Cannot cast an expression with the void sentinel type";
  // const fold IntImm as they are used in index computations
  if (dtype.IsScalar()) {
    if (const IntImmNode* op = value.as<IntImmNode>()) {
      return MakeConst(dtype, op->value, op->span);
    } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
      return MakeConst(dtype, op->value, op->span);
    }
    return prim::Cast(std::move(t), value, span);
  } else {
    PrimType elem_ty = dtype.WithLanes(1);
    if (!value.ty().IsScalableVector() && !value.ty().IsFixedLengthVector()) {
      // manually unroll cast
      if (value.ty() != elem_ty) {
        if (const IntImmNode* op = value.as<IntImmNode>()) {
          value = MakeConst(elem_ty, op->value, op->span);
        } else if (const FloatImmNode* op = value.as<FloatImmNode>()) {
          value = MakeConst(elem_ty, op->value, op->span);
        } else {
          value = prim::Cast(elem_ty, value, span);
        }
      }
      if (dtype.IsScalableVector()) {
        return prim::Broadcast(
            value,
            prim::Mul(dtype.VScaleFactor(),
                      Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>()),
            span);
      } else {
        return prim::Broadcast(value, dtype.lanes(), span);
      }
    } else { /* value is a vector */
      TVM_FFI_ICHECK(value.ty().IsScalableVector() == dtype.IsScalableVector());

      bool lanes_match = false;
      if (value.ty().IsScalableVector()) {
        lanes_match = value.ty().VScaleFactor() == dtype.VScaleFactor();
      } else {
        lanes_match = value.ty().lanes() == dtype.lanes();
      }
      TVM_FFI_ICHECK(lanes_match);
      if (const auto* broadcast = value.as<prim::BroadcastNode>()) {
        return prim::Broadcast(cast(elem_ty, broadcast->value, span), broadcast->lanes, span);
      } else if (const auto* ramp = value.as<prim::RampNode>()) {
        if (dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
          // only cast to index data type can be folded to ramp
          return prim::Ramp(cast(elem_ty, ramp->base, span), cast(elem_ty, ramp->stride, span),
                            ramp->lanes, span);
        }
      }
      return prim::Cast(std::move(t), value, span);
    }
  }
}

PrimExpr cast(DLDataType dtype, PrimExpr value, Span span) {
  return cast(PrimType(dtype), std::move(value), std::move(span));
}

template <typename ValueType>
inline bool ConstPowerHelper(ValueType val, int* shift) {
  if (val <= 0) return false;
  shift[0] = 0;
  while (val != 0) {
    if (val & 1) {
      return (val == 1);
    }
    ++shift[0];
    val = val >> 1;
  }
  return true;
}

bool is_const_power_of_two_integer(const PrimExpr& x, int* shift) {
  if (const auto* op = x.as<IntImmNode>()) {
    return ConstPowerHelper(op->value, shift);
  } else {
    return false;
  }
}

}  // namespace prim

PrimExpr operator+(PrimExpr a, PrimExpr b) { return add(a, b); }

PrimExpr add(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::Add>(a, b)) return ret.value();
  return prim::Add(a, b, span);
}

// negation
PrimExpr operator-(PrimExpr a) { return neg(a); }

PrimExpr neg(PrimExpr a, Span span) {
  const IntImmNode* pa = a.as<IntImmNode>();
  const FloatImmNode* fa = a.as<FloatImmNode>();
  if (pa) {
    ffi::BigInt value = -pa->value;
    if (a.ty().MatchesCode(DLDataTypeCode::kDLInt) && a.ty().bits() >= 64) {
      value = prim::detail::GetFoldResult(std::move(value), a.ty());
    }
    return IntImm(a.ty(), std::move(value), span);
  }
  if (fa) return FloatImm(a.ty(), -fa->value, span);
  return MakeConst(a.ty(), 0, span) - a;
}

PrimExpr operator-(PrimExpr a, PrimExpr b) { return sub(a, b); }

PrimExpr sub(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::Sub>(a, b)) return ret.value();
  return prim::Sub(a, b, span);
}

PrimExpr operator*(PrimExpr a, PrimExpr b) { return mul(a, b); }
PrimExpr mul(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::Mul>(a, b)) return ret.value();
  return prim::Mul(a, b, span);
}

PrimExpr div(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::Div>(a, b)) return ret.value();
  return prim::Div(a, b, span);
}

PrimExpr truncdiv(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << b;
  return div(a, b, span);
}

PrimExpr truncmod(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::Mod>(a, b)) return ret.value();
  return prim::Mod(a, b, span);
}

PrimExpr operator/(PrimExpr a, PrimExpr b) { return div(a, b); }

PrimExpr operator%(PrimExpr a, PrimExpr b) { return truncmod(a, b); }

// TODO(tqchen): switch to floordiv
PrimExpr indexdiv(PrimExpr a, PrimExpr b, Span span) { return floordiv(a, b, span); }

PrimExpr shapediv(PrimExpr a, PrimExpr b, Span span) { return ceildiv(a, b, span); }

PrimExpr indexmod(PrimExpr a, PrimExpr b, Span span) { return floormod(a, b, span); }

PrimExpr floordiv(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::FloorDiv>(a, b)) return ret.value();
  return prim::FloorDiv(a, b, span);
}

PrimExpr ceildiv(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::FloorDiv>(a + b - 1, b)) return ret.value();
  return prim::FloorDiv(a + b - 1, b, span);
}

PrimExpr floormod(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::FloorMod>(a, b)) return ret.value();
  return prim::FloorMod(a, b, span);
}

PrimExpr min(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using prim::detail::is_neg_inf;
  using prim::detail::is_pos_inf;
  if (is_pos_inf(a)) return b;
  if (is_neg_inf(a)) return a;
  if (is_pos_inf(b)) return a;
  if (is_neg_inf(b)) return b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::Min>(a, b)) return ret.value();
  return prim::Min(a, b, span);
}

PrimExpr max(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using prim::detail::is_neg_inf;
  using prim::detail::is_pos_inf;
  if (is_pos_inf(a)) return a;
  if (is_neg_inf(a)) return b;
  if (is_pos_inf(b)) return b;
  if (is_neg_inf(b)) return a;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::Max>(a, b)) return ret.value();
  return prim::Max(a, b, span);
}

// if_then_else
PrimExpr if_then_else(PrimExpr cond, PrimExpr true_value, PrimExpr false_value, Span span) {
  TVM_FFI_ICHECK(cond.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "if_then_else only accept the condition to be boolean type.";
  BinaryOpMatchTypes(true_value, false_value, span);
  if (const IntImmNode* op = cond.as<IntImmNode>()) {
    if (op->value != 0) {
      return true_value;
    } else {
      return false_value;
    }
  }

  return Call(true_value.ty(), prim::builtin::if_then_else(), {cond, true_value, false_value}, {},
              {}, span)
      .as_or_throw<PrimExpr>();
}

// likely
PrimExpr likely(PrimExpr cond, Span span) {
  if (is_const_int(cond)) return cond;
  return Call(cond.ty(), prim::builtin::likely(), {cond}, {}, {}, span).as_or_throw<PrimExpr>();
}

// operator>
PrimExpr operator>(PrimExpr a, PrimExpr b) { return greater(a, b); }
PrimExpr greater(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::GT>(a, b)) return ret.value();
  return prim::GT(a, b, span);
}

PrimExpr operator>=(PrimExpr a, PrimExpr b) { return greater_equal(a, b); }
PrimExpr greater_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::GE>(a, b)) return ret.value();
  return prim::GE(a, b, span);
}

PrimExpr operator<(PrimExpr a, PrimExpr b) { return less(a, b); }
PrimExpr less(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::LT>(a, b)) return ret.value();
  return prim::LT(a, b, span);
}

PrimExpr operator<=(PrimExpr a, PrimExpr b) { return less_equal(a, b); }
PrimExpr less_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::LE>(a, b)) return ret.value();
  return prim::LE(a, b, span);
}

PrimExpr operator==(PrimExpr a, PrimExpr b) { return equal(a, b); }
PrimExpr equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::EQ>(a, b)) return ret.value();
  if (IsVScaleCall(a) && IsVScaleCall(b)) return true;
  return prim::EQ(a, b, span);
}

PrimExpr operator!=(PrimExpr a, PrimExpr b) { return not_equal(a, b); }
PrimExpr not_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = prim::detail::TryConstFold<prim::NE>(a, b)) return ret.value();
  return prim::NE(a, b, span);
}

PrimExpr operator&&(PrimExpr a, PrimExpr b) { return logical_and(a, b); }
PrimExpr logical_and(PrimExpr a, PrimExpr b, Span span) {
  type_check_boolean_args(a, b, "&& operator (logical AND)");
  if (auto ret = prim::detail::TryConstFold<prim::And>(a, b)) return ret.value();
  return prim::And(a, b, span);
}

PrimExpr operator||(PrimExpr a, PrimExpr b) { return logical_or(a, b); }
PrimExpr logical_or(PrimExpr a, PrimExpr b, Span span) {
  type_check_boolean_args(a, b, "|| operator (logical OR)");
  if (auto ret = prim::detail::TryConstFold<prim::Or>(a, b)) return ret.value();
  return prim::Or(a, b, span);
}

PrimExpr operator!(PrimExpr a) { return logical_not(a); }
PrimExpr logical_not(PrimExpr a, Span span) {
  type_check_boolean_args(a, "! operator (logical NOT)");
  if (auto ret = prim::detail::TryConstFold<prim::Not>(a)) return ret.value();
  return prim::Not(a, span);
}

// shift right
PrimExpr operator>>(PrimExpr a, PrimExpr b) { return right_shift(a, b); }

PrimExpr right_shift(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, ">> operator (right shift)");

  BinaryOpMatchTypes(a, b, span);
  TVM_PRIM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pb)
      TVM_FFI_ICHECK(pb->value >= 0 && pb->value < result_ty.bits())
          << "Shift amount must be non-negative and less than " << result_ty.bits() << " for type "
          << result_ty;
    if (pa && pb) {
      return IntImm(result_ty, (pa->value >> pb->value), span);
    }
    if (pb) {
      if (pb->value == 0) return a;
    }
  });

  return Call(a.ty(), prim::builtin::shift_right(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// shift left
PrimExpr operator<<(PrimExpr a, PrimExpr b) { return left_shift(a, b); }
PrimExpr left_shift(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, "<< operator (left shift)");
  BinaryOpMatchTypes(a, b, span);
  TVM_PRIM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pb)
      TVM_FFI_ICHECK(pb->value >= 0 && pb->value < result_ty.bits())
          << "Shift amount must be non-negative and less than " << result_ty.bits() << " for type "
          << result_ty;
    if (pa && pb) {
      ffi::BigInt value = pa->value << pb->value;
      if (result_ty.bits() >= 64) value = prim::detail::GetFoldResult(std::move(value), result_ty);
      return IntImm(result_ty, std::move(value), span);
    }
    if (pb) {
      if (pb->value == 0) return a;
    }
  });
  return Call(a.ty(), prim::builtin::shift_left(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise and
PrimExpr operator&(PrimExpr a, PrimExpr b) { return bitwise_and(a, b); }
PrimExpr bitwise_and(PrimExpr a, PrimExpr b, Span span) {
  type_check_int_or_bool_args(a, b, "& operator (bitwise AND)");
  BinaryOpMatchTypes(a, b, span);
  TVM_PRIM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pa && pb) return IntImm(result_ty, (pa->value & pb->value), span);
  });
  return Call(a.ty(), prim::builtin::bitwise_and(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise_or
PrimExpr operator|(PrimExpr a, PrimExpr b) { return bitwise_or(a, b); }
PrimExpr bitwise_or(PrimExpr a, PrimExpr b, Span span) {
  type_check_int_or_bool_args(a, b, "| operator (bitwise OR)");
  BinaryOpMatchTypes(a, b, span);
  TVM_PRIM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pa && pb) return IntImm(result_ty, (pa->value | pb->value), span);
  });
  return Call(a.ty(), prim::builtin::bitwise_or(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise_xor
PrimExpr operator^(PrimExpr a, PrimExpr b) { return bitwise_xor(a, b); }
PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, Span span) {
  type_check_int_or_bool_args(a, b, "^ operator (bitwise XOR)");
  BinaryOpMatchTypes(a, b, span);
  TVM_PRIM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pa && pb) return IntImm(result_ty, (pa->value ^ pb->value), span);
  });
  return Call(a.ty(), prim::builtin::bitwise_xor(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise_not
PrimExpr operator~(PrimExpr a) { return bitwise_neg(a); }

PrimExpr bitwise_neg(PrimExpr a, Span span) {
  type_check_int_or_bool_args(a, "~ operator (bitwise NOT)");
  return Call(a.ty(), prim::builtin::bitwise_not(), {a}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("prim.bitwise_not",
                        [](PrimExpr a, Span span) { return bitwise_neg(a, span); });
}

PrimExpr prim::IntegerAbs(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    if (const IntImmNode* px = x.as<IntImmNode>()) {
      ffi::BigInt value = px->value < 0 ? -px->value : px->value;
      if (x.ty().bits() >= 64) value = prim::detail::GetFoldResult(std::move(value), x.ty());
      return IntImm(x.ty(), std::move(value), px->span);
    }
    return Select(x >= MakeConst(x.ty(), 0), x, -x, span);
  }
  if (x.ty().MatchesCode(DLDataTypeCode::kDLUInt)) return x;
  TVM_FFI_THROW(InternalError) << "Integer absolute value requires an integer type, got " << x.ty();
}

// operator overloading, smarter than make
#define DEF_MAKE_BINARY_OP(Node, Func) \
  def("prim." #Node, [](PrimExpr a, PrimExpr b, Span span) { return (Func(a, b, span)); })

#define DEF_MAKE_BIT_OP(Node, Func)                                                            \
  def_packed("prim." #Node, [](ffi::PackedArgs args, ffi::Any* ret) {                          \
    bool lhs_is_int = args[0].type_index() == ffi::TypeIndex::kTVMFFIInt;                      \
    bool rhs_is_int = args[1].type_index() == ffi::TypeIndex::kTVMFFIInt;                      \
    if (lhs_is_int) {                                                                          \
      *ret = (Func(args[0].cast<int>(), args[1].cast<PrimExpr>(), args[2].cast<Span>()));      \
    } else if (rhs_is_int) {                                                                   \
      *ret = (Func(args[0].cast<PrimExpr>(), args[1].cast<int>(), args[2].cast<Span>()));      \
    } else {                                                                                   \
      *ret = (Func(args[0].cast<PrimExpr>(), args[1].cast<PrimExpr>(), args[2].cast<Span>())); \
    }                                                                                          \
  })

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("prim._OpIfThenElse",
           [](PrimExpr cond, PrimExpr true_value, PrimExpr false_value, Span span) {
             return if_then_else(cond, true_value, false_value, span);
           })
      .DEF_MAKE_BINARY_OP(_OpAdd, add)
      .DEF_MAKE_BINARY_OP(_OpSub, sub)
      .DEF_MAKE_BINARY_OP(_OpMul, mul)
      .DEF_MAKE_BINARY_OP(_OpDiv, div)
      .DEF_MAKE_BINARY_OP(_OpMod, truncmod)
      .DEF_MAKE_BINARY_OP(_OpIndexDiv, indexdiv)
      .DEF_MAKE_BINARY_OP(_OpIndexMod, indexmod)
      .DEF_MAKE_BINARY_OP(_OpFloorDiv, floordiv)
      .DEF_MAKE_BINARY_OP(_OpFloorMod, floormod)
      .DEF_MAKE_BINARY_OP(_OpTruncDiv, truncdiv)
      .DEF_MAKE_BINARY_OP(_OpTruncMod, truncmod)
      .DEF_MAKE_BINARY_OP(_OpCeilDiv, ceildiv)
      .DEF_MAKE_BINARY_OP(_OpMin, min)
      .DEF_MAKE_BINARY_OP(_OpMax, max)
      .DEF_MAKE_BINARY_OP(_OpEQ, equal)
      .DEF_MAKE_BINARY_OP(_OpNE, not_equal)
      .DEF_MAKE_BINARY_OP(_OpLT, less)        // NOLINT(*)
      .DEF_MAKE_BINARY_OP(_OpLE, less_equal)  // NOLINT(*)
      .DEF_MAKE_BINARY_OP(_OpGT, greater)     // NOLINT(*)
      .DEF_MAKE_BINARY_OP(_OpGE, greater_equal)
      .DEF_MAKE_BINARY_OP(_OpAnd, logical_and)
      .DEF_MAKE_BINARY_OP(_OpOr, logical_or)
      .DEF_MAKE_BIT_OP(bitwise_and, bitwise_and)
      .DEF_MAKE_BIT_OP(bitwise_or, bitwise_or)
      .DEF_MAKE_BIT_OP(bitwise_xor, bitwise_xor)
      .DEF_MAKE_BIT_OP(left_shift, left_shift)  // NOLINT(*)
      .DEF_MAKE_BIT_OP(right_shift, right_shift);
}

// ceil
PrimExpr ceil(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::ceil(fx->value), fx->span);
  return Call(x.ty(), prim::builtin::ceil(), {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

PrimExpr log2(PrimExpr x, Span span) {
  PrimType x_ty = x.ty();
  if (x_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    PrimType f32_ty = x_ty.IsScalableVector() ? PrimType::ScalableVector(DLDataTypeCode::kDLFloat,
                                                                         32, x_ty.VScaleFactor())
                                              : PrimType::Float(32, x_ty.lanes());
    PrimExpr x_fp32 = prim::Cast(f32_ty, x, span);
    PrimExpr result_fp32 =
        Call(f32_ty, prim::builtin::log2(), {x_fp32}, {}, {}, span).as_or_throw<PrimExpr>();
    return prim::Cast(x_ty, result_fp32, span);
  }
  return Call(x_ty, prim::builtin::log2(), {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

PrimExpr prim::clz(PrimExpr x, Span span) {
  PrimType x_ty = x.ty();
  if (x_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    PrimType f32_ty = x_ty.IsScalableVector() ? PrimType::ScalableVector(DLDataTypeCode::kDLFloat,
                                                                         32, x_ty.VScaleFactor())
                                              : PrimType::Float(32, x_ty.lanes());
    PrimExpr x_fp32 = prim::Cast(f32_ty, x, span);
    PrimExpr result_fp32 =
        Call(f32_ty, prim::builtin::clz(), {x_fp32}, {}, {}, span).as_or_throw<PrimExpr>();
    return prim::Cast(x_ty, result_fp32, span);
  }
  return Call(x_ty, prim::builtin::clz(), {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef()
      .def_packed("node._const",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    if (auto opt = args[0].try_cast<ffi::BigInt>(); opt.has_value()) {
                      *ret = prim::MakeConst(args[1].cast<PrimType>(), *opt, args[2].cast<Span>());
                    } else if (auto opt = args[0].try_cast<double>()) {
                      *ret = prim::MakeConst(args[1].cast<PrimType>(), *opt, args[2].cast<Span>());
                    } else {
                      TVM_FFI_THROW(InternalError)
                          << "First argument to tvm.tirx.const must be int, float, or bool, "
                          << "but instead received argument with type code "
                          << args[0].GetTypeKey();
                    }
                  })
      .def("prim.max_value", static_cast<PrimExpr (*)(PrimType, Span)>(&prim::max_value))
      .def("prim._cast",
           [](PrimType dtype, PrimExpr value, Span span) { return prim::cast(dtype, value, span); })
      .def("prim.min_value", static_cast<PrimExpr (*)(PrimType, Span)>(&prim::min_value))
      .def("prim.likely", tvm::likely)
      .def("prim.ceil", tvm::ceil);
}

}  // namespace tvm
