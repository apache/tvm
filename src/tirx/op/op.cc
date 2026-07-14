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
 * \file tirx/op/op.cc
 *
 *  Common operator definitions for ops in tirx/op.h
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <cmath>
// Centralized header for constant folders.
#include "../../arith/const_fold.h"
#include "../analysis/check_contains.h"

namespace tvm {

using namespace tirx;

namespace {
// File-local helper: true if `expr` is a call to tirx::builtin::vscale().
bool IsVScaleCall(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallNode>()) {
    return call->op.same_as(builtin::vscale());
  }
  return false;
}

TVM_FFI_INLINE const PrimTypeNode* GetPrimTypeNode(const PrimExpr& expr) {
  // Avoid PrimExpr::ty() ObjectRef materialization on binary operator hot paths.
  const auto* node = expr.get();
  TVM_FFI_DCHECK(node != nullptr);
  TVM_FFI_DCHECK(!node->ExprNode::ty.IsMissing());
  const auto* prim_ty = node->ExprNode::ty.as<PrimTypeNode>();
  TVM_FFI_DCHECK(prim_ty != nullptr);
  return prim_ty;
}

TVM_FFI_INLINE bool IsFloatType(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat);
}

TVM_FFI_INLINE bool IsBFloat16Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLBfloat);
}

TVM_FFI_INLINE bool IsFloat8Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3,
                        DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn,
                        DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2,
                        DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu);
}

TVM_FFI_INLINE bool IsFloat6Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat6_e2m3fn, DLDataTypeCode::kDLFloat6_e3m2fn);
}

TVM_FFI_INLINE bool IsFloat4Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn);
}
}  // namespace

// macro to register an unary op
#define TVM_TIR_REGISTER_PURE_UNARY_OP(OpName)                             \
  TVM_TIR_REGISTER_OP(OpName).set_num_inputs(1).set_attr<TCallEffectKind>( \
      "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))

// macro to register an binary op
#define TVM_TIR_REGISTER_PURE_BINARY_OP(OpName)                            \
  TVM_TIR_REGISTER_OP(OpName).set_num_inputs(2).set_attr<TCallEffectKind>( \
      "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))

Type GetType(const PrimExpr& expr) {
  // TODO(tqchen): add recursive type inference for Call here
  // once we introduced the corresponding fields to the IR.
  if (auto* ptr = expr.as<tirx::VarNode>()) {
    // If Var has a more refined type annotation,
    // return the type anotation
    if (!ptr->ty.IsMissing()) {
      return ptr->ty;
    }
  }

  static const Op& type_annotation_op = Op::Get("tirx.type_annotation");
  if (auto* access = expr.as<CallNode>()) {
    if (access->op.same_as(builtin::tvm_access_ptr())) {
      TVM_FFI_ICHECK(access->args.size())
          << "Builtin tvm_access_ptr() may not have empty arguments";
      auto type_annotation = access->args[0].as_or_throw<Call>();
      TVM_FFI_ICHECK(type_annotation->op.same_as(type_annotation_op))
          << "Expected the first argument of builtin tvm_access_ptr() "
          << "to be a type annotation, but found " << type_annotation->op;
      return PointerType(type_annotation->ty.as_or_throw<PrimType>());
    }
    if (access->op.same_as(builtin::ptr_byte_offset())) {
      TVM_FFI_ICHECK_EQ(access->args.size(), 3U);
      auto type_annotation = access->args[2].as_or_throw<Call>();
      TVM_FFI_ICHECK(type_annotation->op.same_as(type_annotation_op))
          << "Expected the third argument of builtin ptr_byte_offset() "
          << "to be a type annotation, but found " << type_annotation->op;
      return PointerType(type_annotation->ty.as_or_throw<PrimType>());
    }
  }

  if (auto* address_of = expr.as<CallNode>()) {
    if (address_of->op.same_as(builtin::address_of())) {
      TVM_FFI_ICHECK_EQ(address_of->args.size(), 1)
          << "Builtin address_of() expects a single argument, but received arguments "
          << address_of->args;
      auto* address = address_of->args[0].as<BufferLoadNode>();
      if (address) {
        return PointerType(address->ty.as_or_throw<PrimType>());
      }

      if (auto* var = address_of->args[0].as<VarNode>()) {
        if (auto* ptr = var->ty.as<PointerTypeNode>()) {
          if (ptr->element_type.as<TensorMapTypeNode>()) {
            return PrimType::UInt(64);
          }
        }
        return PointerType(var->ty.as_or_throw<PrimType>());
      }

      TVM_FFI_ICHECK(false)
          << "Builtin address_of() expects the argument to be a BufferLoad or Var, but "
          << "received argument " << address_of->args[0];
    }
  }
  return expr.ty();
}

Type GetTypeFromRuntimeDataType(DLDataType dtype) {
  if (dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLOpaqueHandle) &&
      (dtype.bits != 0 || dtype.lanes != 0)) {
    return PointerType::VoidPointerTy();
  }
  return PrimType(dtype);
}

// LargeUIntImm
PrimExpr LargeUIntImm(PrimType value_ty, int64_t low, int64_t high, Span span) {
  return Call(value_ty, tirx::builtin::large_uint_imm(),
              {IntImm(PrimType::UInt(32), low, span), IntImm(PrimType::UInt(32), high, span)}, {},
              {}, span)
      .as_or_throw<PrimExpr>();
}

// Q-multiplication
PrimExpr q_multiply_shift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr s, Span span) {
  return Call(PrimType::Int(32, x.ty().lanes()), tirx::builtin::q_multiply_shift(), {x, y, q, s},
              {}, {}, span)
      .as_or_throw<PrimExpr>();
}

void BroadcastToMatchLanes(PrimExpr& op_a, PrimExpr& op_b) {  // NOLINT(*)
  PrimType ty_a = op_a.ty();
  PrimType ty_b = op_b.ty();

  if (!ty_a.IsScalableVector() && !ty_a.IsFixedLengthVector() &&
      (ty_b.IsScalableVector() || ty_b.IsFixedLengthVector())) {
    if (ty_b.IsScalableVector()) {
      PrimType i32_ty = PrimType::Int(32);
      op_a = tirx::Broadcast(
          op_a, tirx::Mul(ty_b.VScaleFactor(),
                          Call(i32_ty, builtin::vscale(), {}).as_or_throw<PrimExpr>()));
    } else {
      op_a = tirx::Broadcast(op_a, ty_b.lanes());
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
    lhs = cast(promoted_ty, lhs, span);
  }
  if (rhs_ty->dtype != promoted_ty->dtype) {
    rhs = cast(promoted_ty, rhs, span);
  }
}

PrimExpr ret(PrimExpr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  return Call(value.ty(), tirx::builtin::ret(), {value}, {}, {}, span).as_or_throw<PrimExpr>();
}

Expr ret(Expr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  if (auto prim_value = value.as<PrimExpr>()) {
    return ret(prim_value.value(), span);
  }
  return Call(value->ty, tirx::builtin::ret(), {value}, {}, {}, span);
}

PrimExpr thread_return(Span span) {
  return Call(PrimType::Void(), tirx::builtin::thread_return(), {}, {}, {}, span)
      .as_or_throw<PrimExpr>();
}

PrimExpr continue_loop(Span span) {
  return Call(PrimType::Void(), tirx::builtin::continue_loop(), {}, {}, {}, span)
      .as_or_throw<PrimExpr>();
}

PrimExpr break_loop(Span span) {
  return Call(PrimType::Void(), tirx::builtin::break_loop(), {}, {}, {}, span)
      .as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.ret", [](Expr value, Span span) { return ret(value, span); })
      .def("tirx.thread_return", thread_return)
      .def("tirx.continue_loop", continue_loop)
      .def("tirx.break_loop", break_loop);
};

// maximum and min limits
PrimExpr max_value(PrimType value_ty, Span span) {
  using namespace tirx;
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
  using namespace tirx;
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

// infinity
PrimExpr infinity(PrimType value_ty, Span span) {
  using namespace tirx;
  PrimType dtype = value_ty;
  TVM_FFI_ICHECK_EQ(dtype.lanes(), 1);
  if (IsFloatType(dtype)) {
    if (dtype.bits() == 64) {
      return FloatImm(value_ty, std::numeric_limits<double>::infinity(), span);
    } else if (dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(value_ty, std::numeric_limits<float>::infinity(), span);
    }
  }
  TVM_FFI_THROW(InternalError) << "Cannot decide infinity for type " << dtype;
}

namespace tirx {
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
  if (const auto* op = x.as<tirx::IntImmNode>()) {
    return ConstPowerHelper(op->value, shift);
  } else {
    return false;
  }
}
}  // namespace tirx

PrimExpr cast(PrimType t, PrimExpr value, Span span) {
  using tirx::FloatImmNode;
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
    return tirx::Cast(std::move(t), value, span);
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
          value = tirx::Cast(elem_ty, value, span);
        }
      }
      if (dtype.IsScalableVector()) {
        return tirx::Broadcast(
            value,
            tirx::Mul(dtype.VScaleFactor(),
                      Call(PrimType::Int(32), builtin::vscale(), {}).as_or_throw<PrimExpr>()),
            span);
      } else {
        return tirx::Broadcast(value, dtype.lanes(), span);
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
      if (const auto* broadcast = value.as<tirx::BroadcastNode>()) {
        return tirx::Broadcast(cast(elem_ty, broadcast->value, span), broadcast->lanes, span);
      } else if (const auto* ramp = value.as<tirx::RampNode>()) {
        if (dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
          // only cast to index data type can be folded to ramp
          return tirx::Ramp(cast(elem_ty, ramp->base, span), cast(elem_ty, ramp->stride, span),
                            ramp->lanes, span);
        }
      }
      return tirx::Cast(std::move(t), value, span);
    }
  }
}

PrimExpr cast(DLDataType t, PrimExpr value, Span span) {
  return cast(PrimType(t), std::move(value), std::move(span));
}

// reinterpret
PrimExpr reinterpret(PrimType t, PrimExpr value, Span span) {
  PrimType target_dtype = t;
  PrimType value_dtype = value.ty();
  if (value.ty() == t) return value;
  if (!target_dtype.IsScalableVector() && !value_dtype.IsScalableVector()) {
    int value_bits = value_dtype.bits() * value_dtype.lanes();
    int target_bits = target_dtype.bits() * target_dtype.lanes();
    TVM_FFI_ICHECK(value_bits == target_bits ||
                   ((value_dtype.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) ||
                     target_dtype.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn)) &&
                    value_dtype.StorageBytes() == target_dtype.StorageBytes()))
        << "Reinterpret requires size match " << target_dtype << " vs " << value_dtype;
  }
  return Call(std::move(t), tirx::builtin::reinterpret(), {value}, {}, {}, span)
      .as_or_throw<PrimExpr>();
}

Expr reinterpret(Type target_ty, Expr value, Span span) {
  if (value.as<StringImmNode>()) {
    TVM_FFI_CHECK(target_ty.as<PointerTypeNode>(), TypeError)
        << "String reinterpret requires a pointer target, but got " << target_ty;
    return Call(std::move(target_ty), tirx::builtin::reinterpret(), {std::move(value)}, {}, {},
                std::move(span));
  }
  if (auto target_dtype = target_ty.as<PrimType>()) {
    if (auto prim_value = value.as<PrimExpr>()) {
      return reinterpret(target_dtype.value(), prim_value.value(), std::move(span));
    }
    TVM_FFI_CHECK(value->ty.as<PointerTypeNode>(), TypeError)
        << "Reinterpret source must be PrimType or PointerType, but got " << value->ty;
    TVM_FFI_CHECK(
        target_dtype.value().IsScalar() && target_dtype.value().bits() == 64 &&
            target_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt),
        TypeError)
        << "Pointer reinterpret requires a scalar 64-bit integer target, but got "
        << target_dtype.value();
  } else {
    TVM_FFI_CHECK(target_ty.as<PointerTypeNode>(), TypeError)
        << "Reinterpret target must be PrimType or PointerType, but got " << target_ty;
    if (auto source_dtype = value->ty.as<PrimType>()) {
      TVM_FFI_CHECK(
          source_dtype.value().IsScalar() && source_dtype.value().bits() == 64 &&
              source_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt),
          TypeError)
          << "Pointer reinterpret requires a scalar 64-bit integer source, but got "
          << source_dtype.value();
    } else {
      TVM_FFI_CHECK(value->ty.as<PointerTypeNode>(), TypeError)
          << "Reinterpret source must be PrimType or PointerType, but got " << value->ty;
    }
  }
  return Call(std::move(target_ty), tirx::builtin::reinterpret(), {std::move(value)}, {}, {},
              std::move(span));
}

PrimExpr reinterpret(DLDataType t, PrimExpr value, Span span) {
  return reinterpret(PrimType(t), std::move(value), std::move(span));
}

// operator+
PrimExpr operator+(PrimExpr a, PrimExpr b) { return add(a, b); }

PrimExpr add(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::Add>(a, b)) return ret.value();
  return tirx::Add(a, b, span);
}

// negation
PrimExpr operator-(PrimExpr a) { return neg(a); }

PrimExpr neg(PrimExpr a, Span span) {
  using tirx::FloatImmNode;
  using tirx::IntImmNode;
  const IntImmNode* pa = a.as<IntImmNode>();
  const FloatImmNode* fa = a.as<FloatImmNode>();
  if (pa) return IntImm(a.ty(), -pa->value, span);
  if (fa) return FloatImm(a.ty(), -fa->value, span);
  return MakeConst(a.ty(), 0, span) - a;
}

PrimExpr operator-(PrimExpr a, PrimExpr b) { return sub(a, b); }

PrimExpr sub(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::Sub>(a, b)) return ret.value();
  return tirx::Sub(a, b, span);
}

PrimExpr operator*(PrimExpr a, PrimExpr b) { return mul(a, b); }
PrimExpr mul(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::Mul>(a, b)) return ret.value();
  return tirx::Mul(a, b, span);
}

PrimExpr div(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::Div>(a, b)) return ret.value();
  return tirx::Div(a, b, span);
}

PrimExpr truncdiv(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << b;
  return div(a, b, span);
}

PrimExpr truncmod(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::Mod>(a, b)) return ret.value();
  return tirx::Mod(a, b, span);
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
  if (auto ret = arith::TryConstFold<tirx::FloorDiv>(a, b)) return ret.value();
  return tirx::FloorDiv(a, b, span);
}

PrimExpr logaddexp(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(IsFloatType(a.ty())) << a;
  TVM_FFI_ICHECK(IsFloatType(b.ty())) << b;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr exp_sum = add(exp(a), exp(b));
  PrimExpr log_exp_sum = log(exp_sum);
  return log_exp_sum;
}

PrimExpr ceildiv(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::FloorDiv>(a + b - 1, b)) return ret.value();
  return tirx::FloorDiv(a + b - 1, b, span);
}

PrimExpr floormod(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) << b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::FloorMod>(a, b)) return ret.value();
  return tirx::FloorMod(a, b, span);
}

PrimExpr min(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using arith::is_neg_inf;
  using arith::is_pos_inf;
  if (is_pos_inf(a)) return b;
  if (is_neg_inf(a)) return a;
  if (is_pos_inf(b)) return a;
  if (is_neg_inf(b)) return b;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::Min>(a, b)) return ret.value();
  return tirx::Min(a, b, span);
}

PrimExpr max(PrimExpr a, PrimExpr b, Span span) {
  // inf-aware simplificaiton
  using arith::is_neg_inf;
  using arith::is_pos_inf;
  if (is_pos_inf(a)) return a;
  if (is_neg_inf(a)) return b;
  if (is_pos_inf(b)) return b;
  if (is_neg_inf(b)) return a;
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::Max>(a, b)) return ret.value();
  return tirx::Max(a, b, span);
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

  return Call(true_value.ty(), tirx::builtin::if_then_else(), {cond, true_value, false_value}, {},
              {}, span)
      .as_or_throw<PrimExpr>();
}

// likely
PrimExpr likely(PrimExpr cond, Span span) {
  if (is_const_int(cond)) return cond;
  return Call(cond.ty(), tirx::builtin::likely(), {cond}, {}, {}, span).as_or_throw<PrimExpr>();
}

// operator>
PrimExpr operator>(PrimExpr a, PrimExpr b) { return greater(a, b); }
PrimExpr greater(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::GT>(a, b)) return ret.value();
  return tirx::GT(a, b, span);
}

PrimExpr operator>=(PrimExpr a, PrimExpr b) { return greater_equal(a, b); }
PrimExpr greater_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::GE>(a, b)) return ret.value();
  return tirx::GE(a, b, span);
}

PrimExpr operator<(PrimExpr a, PrimExpr b) { return less(a, b); }
PrimExpr less(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::LT>(a, b)) return ret.value();
  return tirx::LT(a, b, span);
}

PrimExpr operator<=(PrimExpr a, PrimExpr b) { return less_equal(a, b); }
PrimExpr less_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::LE>(a, b)) return ret.value();
  return tirx::LE(a, b, span);
}

PrimExpr operator==(PrimExpr a, PrimExpr b) { return equal(a, b); }
PrimExpr equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::EQ>(a, b)) return ret.value();
  if (IsVScaleCall(a) && IsVScaleCall(b)) return true;
  return tirx::EQ(a, b, span);
}

PrimExpr operator!=(PrimExpr a, PrimExpr b) { return not_equal(a, b); }
PrimExpr not_equal(PrimExpr a, PrimExpr b, Span span) {
  BinaryOpMatchTypes(a, b, span);
  if (auto ret = arith::TryConstFold<tirx::NE>(a, b)) return ret.value();
  return tirx::NE(a, b, span);
}

namespace {
void type_check_boolean_args(const PrimExpr& arg, const char* op) {
  TVM_FFI_ICHECK(arg.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument for " << op << ", but received " << arg << " of type "
      << arg.ty();
}
void type_check_boolean_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  TVM_FFI_ICHECK(lhs.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument as LHS of " << op << ", but received " << lhs << " of type "
      << lhs.ty();
  TVM_FFI_ICHECK(rhs.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument as RHS of " << op << ", but received " << rhs << " of type "
      << rhs.ty();
}

void type_check_int_or_bool_args(const PrimExpr& arg, const char* op) {
  TVM_FFI_ICHECK(arg.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                      DLDataTypeCode::kDLBool))
      << "Expected integer or boolean argument for " << op << ", but received " << arg
      << " of type " << arg.ty();
}

void type_check_integer_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  TVM_FFI_ICHECK(lhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))
      << "Expected integer argument as LHS of " << op << ", but received " << lhs << " of type "
      << lhs.ty();
  TVM_FFI_ICHECK(rhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))
      << "Expected integer argument as RHS of " << op << ", but received " << rhs << " of type "
      << rhs.ty();
}

void type_check_int_or_bool_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  TVM_FFI_ICHECK(lhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                      DLDataTypeCode::kDLBool))
      << "Expected integer argument as LHS of " << op << ", but received " << lhs << " of type "
      << lhs.ty();
  TVM_FFI_ICHECK(rhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                      DLDataTypeCode::kDLBool))
      << "Expected integer argument as RHS of " << op << ", but received " << rhs << " of type "
      << rhs.ty();
}
}  // namespace

PrimExpr operator&&(PrimExpr a, PrimExpr b) { return logical_and(a, b); }
PrimExpr logical_and(PrimExpr a, PrimExpr b, Span span) {
  type_check_boolean_args(a, b, "&& operator (logical AND)");
  if (auto ret = arith::TryConstFold<tirx::And>(a, b)) return ret.value();
  return tirx::And(a, b, span);
}

PrimExpr operator||(PrimExpr a, PrimExpr b) { return logical_or(a, b); }
PrimExpr logical_or(PrimExpr a, PrimExpr b, Span span) {
  type_check_boolean_args(a, b, "|| operator (logical OR)");
  if (auto ret = arith::TryConstFold<tirx::Or>(a, b)) return ret.value();
  return tirx::Or(a, b, span);
}

PrimExpr operator!(PrimExpr a) { return logical_not(a); }
PrimExpr logical_not(PrimExpr a, Span span) {
  type_check_boolean_args(a, "! operator (logical NOT)");
  if (auto ret = arith::TryConstFold<tirx::Not>(a)) return ret.value();
  return tirx::Not(a, span);
}

// shift right
PrimExpr operator>>(PrimExpr a, PrimExpr b) { return right_shift(a, b); }

PrimExpr right_shift(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, ">> operator (right shift)");

  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
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

  return Call(a.ty(), tirx::builtin::shift_right(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// shift left
PrimExpr operator<<(PrimExpr a, PrimExpr b) { return left_shift(a, b); }
PrimExpr left_shift(PrimExpr a, PrimExpr b, Span span) {
  type_check_integer_args(a, b, "<< operator (left shift)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pb)
      TVM_FFI_ICHECK(pb->value >= 0 && pb->value < result_ty.bits())
          << "Shift amount must be non-negative and less than " << result_ty.bits() << " for type "
          << result_ty;
    if (pa && pb) return IntImm(result_ty, (pa->value << pb->value), span);
    if (pb) {
      if (pb->value == 0) return a;
    }
  });
  return Call(a.ty(), tirx::builtin::shift_left(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise and
PrimExpr operator&(PrimExpr a, PrimExpr b) { return bitwise_and(a, b); }
PrimExpr bitwise_and(PrimExpr a, PrimExpr b, Span span) {
  type_check_int_or_bool_args(a, b, "& operator (bitwise AND)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pa && pb) return IntImm(result_ty, (pa->value & pb->value), span);
  });
  return Call(a.ty(), tirx::builtin::bitwise_and(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise_or
PrimExpr operator|(PrimExpr a, PrimExpr b) { return bitwise_or(a, b); }
PrimExpr bitwise_or(PrimExpr a, PrimExpr b, Span span) {
  type_check_int_or_bool_args(a, b, "| operator (bitwise OR)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pa && pb) return IntImm(result_ty, (pa->value | pb->value), span);
  });
  return Call(a.ty(), tirx::builtin::bitwise_or(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise_xor
PrimExpr operator^(PrimExpr a, PrimExpr b) { return bitwise_xor(a, b); }
PrimExpr bitwise_xor(PrimExpr a, PrimExpr b, Span span) {
  type_check_int_or_bool_args(a, b, "^ operator (bitwise XOR)");
  BinaryOpMatchTypes(a, b, span);
  TVM_INDEX_CONST_PROPAGATION({
    PrimType result_ty = a.ty();
    if (pa && pb) return IntImm(result_ty, (pa->value ^ pb->value), span);
  });
  return Call(a.ty(), tirx::builtin::bitwise_xor(), {a, b}, {}, {}, span).as_or_throw<PrimExpr>();
}

// bitwise_not
PrimExpr operator~(PrimExpr a) { return bitwise_neg(a); }

PrimExpr bitwise_neg(PrimExpr a, Span span) {
  type_check_int_or_bool_args(a, "~ operator (bitwise NOT)");
  return Call(a.ty(), tirx::builtin::bitwise_not(), {a}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.bitwise_not",
                        [](PrimExpr a, Span span) { return bitwise_neg(a, span); });
}

// pow
PrimExpr pow(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
  TVM_FFI_ICHECK(IsFloatType(x.ty())) << "power only applies to float";

  // If we detect pow(x, 3), suggest using x * x * x
  if (y.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    using tirx::IntImmNode;
    const IntImmNode* px = y.as<IntImmNode>();
    if (px) {
      if (px->value >= 3) {
        LOG(WARNING)
            << "Detected pow(x, y) where y >= 3, it is recommended to avoid this as it may lead to "
               "uninteded behaviors when x < 0. Perhaps with `x * x * x ...` or "
               "`pow(x, 2) * pow(x, 2) ...`.";
      }
    }
  } else if (IsFloatType(y.ty())) {
    using tirx::FloatImmNode;
    const FloatImmNode* fx = y.as<FloatImmNode>();
    if (fx) {
      if (fx->value >= 3.0) {
        LOG(WARNING)
            << "Detected pow(x, y) where y >= 3, it is recommended to avoid this as it may lead to "
               "uninteded behaviors when x < 0. Perhaps with `x * x * x ...` or "
               "`pow(x, 2) * pow(x, 2) ...`.";
      }
    }
  }

  static const Op& pow_op = Op::Get("tirx.pow");
  return Call(x.ty(), pow_op, {x, y}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIR_REGISTER_PURE_BINARY_OP("pow").set_attr<TVectorizable>("TVectorizable", true);

// abs
PrimExpr abs(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    using tirx::IntImmNode;
    const IntImmNode* px = x.as<IntImmNode>();
    if (px) {
      return IntImm(x.ty(), std::abs(px->value), px->span);
    }
    // MakeConst can handle both vector and scalar types.
    return tirx::Select(x >= MakeConst(x.ty(), 0), x, -x, span);
  } else if (IsFloatType(x.ty()) || IsBFloat16Type(x.ty())) {
    using tirx::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return FloatImm(x.ty(), std::fabs(fx->value), fx->span);
    }
    static const Op& fabs_op = Op::Get("tirx.fabs");
    return Call(x.ty(), fabs_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLUInt)) {
    return x;
  } else {
    TVM_FFI_THROW(InternalError) << "Data type " << x.ty()
                                 << " not supported for absolute op. Skipping absolute op...";
    return x;
  }
}

TVM_TIR_REGISTER_PURE_UNARY_OP("fabs").set_attr<TVectorizable>("TVectorizable", true);

// isnan
PrimExpr isnan(PrimExpr x, Span span) {
  PrimType t = PrimType::Bool(x.ty().lanes());
  PrimType bool_ty(t);
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return MakeConst(t, false);
  } else if (IsFloatType(x.ty())) {
    using tirx::FloatImmNode;
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return MakeConst(t, std::isnan(fx->value), fx->span);
    }
    if (x.ty().bits() == 16) {
      static const Op& isnan_op = Op::Get("tirx.isnan");
      PrimType f32_ty = PrimType::Float(32, t.lanes());
      return Call(bool_ty, isnan_op, {cast(f32_ty, std::move(x), span)}, {}, {}, span)
          .as_or_throw<PrimExpr>();
    } else {
      static const Op& isnan_op = Op::Get("tirx.isnan");
      return Call(bool_ty, isnan_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
    }
  } else {
    TVM_FFI_THROW(InternalError) << "Data type " << x.ty()
                                 << " not supported for isnan op. Skipping isnan op...";
  }
}

// isinf
PrimExpr isinf(PrimExpr x, Span span) {
  PrimType t = PrimType::Bool(x.ty().lanes());
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return MakeConst(t, false, span);
  } else if (IsFloatType(x.ty())) {
    PrimExpr infX = infinity(x.ty(), span);
    return abs(x, span) == infX && !isnan(x, span);
  } else {
    TVM_FFI_THROW(InternalError) << "Data type " << x.ty()
                                 << " not supported for finiteness ops. Skipping it...";
  }
}

// isfinite
PrimExpr isfinite(PrimExpr x, Span span) { return !isinf(x, span) && !isnan(x, span); }

PrimExpr sum(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = tirx::Add(x, y, span);
  PrimExpr identity_element = MakeConst(source.ty(), 0, span);
  tirx::CommReducer combiner = tirx::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tirx::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr all(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  type_check_boolean_args(source, "tvm::all");
  PrimVar x("x", source.ty(), span), y("y", source.ty());
  PrimExpr result = tirx::And(x, y, span);
  PrimExpr identity_element = MakeConst(source.ty(), true, span);
  tirx::CommReducer combiner = tirx::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tirx::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr any(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  type_check_boolean_args(source, "tvm::any");
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = tirx::Or(x, y, span);
  PrimExpr identity_element = MakeConst(source.ty(), false, span);
  tirx::CommReducer combiner = tirx::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tirx::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr max(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = tirx::Max(x, y, span);
  PrimExpr identity_element = min_value(source.ty(), span);
  tirx::CommReducer combiner = tirx::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tirx::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr min(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
  PrimExpr result = tirx::Min(x, y, span);
  PrimExpr identity_element = max_value(source.ty(), span);
  tirx::CommReducer combiner = tirx::CommReducer({x}, {y}, {result}, {identity_element}, span);
  return tirx::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
}

PrimExpr prod(PrimExpr source, ffi::Array<IterVar> rdom, ffi::Array<PrimExpr> init, Span span) {
  if (source.ty().MatchesCode(DLDataTypeCode::kDLBool)) {
    // Bool product (prod) has the same truth table as logical AND.  Reuse all() to
    // avoid lowering bool prod through Mul, which LLVM codegen does not support.
    return all(source, rdom, init, span);
  } else {
    // For non-bool types, we lower prod through Mul.
    PrimVar x("x", source.ty(), span), y("y", source.ty(), span);
    PrimExpr result = tirx::Mul(x, y, span);
    PrimExpr identity_element = MakeConst(source.ty(), 1, span);
    tirx::CommReducer combiner = tirx::CommReducer({x}, {y}, {result}, {identity_element}, span);
    return tirx::Reduce(combiner, {source}, rdom, IntImm::Bool(true), 0, init, span);
  }
}

// fmod
PrimExpr fmod(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
  TVM_FFI_ICHECK(IsFloatType(x.ty())) << "fmod only applies to float";
  static const Op& fmod_op = Op::Get("tirx.fmod");
  return Call(x.ty(), fmod_op, {x, y}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIR_REGISTER_PURE_UNARY_OP("fmod");

// floor
PrimExpr floor(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  using tirx::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::floor(fx->value), fx->span);
  static const Op& floor_op = Op::Get("tirx.floor");
  return Call(x.ty(), floor_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIR_REGISTER_PURE_UNARY_OP("floor").set_attr<TVectorizable>("TVectorizable", true);

// ceil
PrimExpr ceil(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  using tirx::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::ceil(fx->value), fx->span);
  static const Op& ceil_op = Op::Get("tirx.ceil");
  return Call(x.ty(), ceil_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIR_REGISTER_PURE_UNARY_OP("ceil").set_attr<TVectorizable>("TVectorizable", true);

// round
PrimExpr round(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  using tirx::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::nearbyint(fx->value), fx->span);
  static const Op& round_op = Op::Get("tirx.round");
  return Call(x.ty(), round_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIR_REGISTER_PURE_UNARY_OP("round").set_attr<TVectorizable>("TVectorizable", true);

// nearbyint
PrimExpr nearbyint(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  using tirx::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::nearbyint(fx->value), fx->span);
  static const Op& nearbyint_op = Op::Get("tirx.nearbyint");
  return Call(x.ty(), nearbyint_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIR_REGISTER_PURE_UNARY_OP("nearbyint");

// trunc
PrimExpr trunc(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  using tirx::FloatImmNode;
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) {
    return FloatImm(x.ty(), (fx->value < 0 ? std::ceil(fx->value) : std::floor(fx->value)),
                    fx->span);
  }
  static const Op& trunc_op = Op::Get("tirx.trunc");
  return Call(x.ty(), trunc_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIR_REGISTER_PURE_UNARY_OP("trunc").set_attr<TVectorizable>("TVectorizable", true);

// unary op registration.
TVM_TIR_REGISTER_PURE_UNARY_OP("exp").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("exp2").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("exp10").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("erf");

TVM_TIR_REGISTER_PURE_UNARY_OP("tanh").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sigmoid").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sqrt").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("rsqrt");

TVM_TIR_REGISTER_PURE_UNARY_OP("log").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("log2").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("log1p");

TVM_TIR_REGISTER_PURE_UNARY_OP("log10").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("tan").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("cos").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("cosh").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sin").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("sinh").set_attr<TVectorizable>("TVectorizable", true);

TVM_TIR_REGISTER_PURE_UNARY_OP("asin");

TVM_TIR_REGISTER_PURE_UNARY_OP("acos");

TVM_TIR_REGISTER_PURE_UNARY_OP("atan");

TVM_TIR_REGISTER_PURE_UNARY_OP("acosh");

TVM_TIR_REGISTER_PURE_UNARY_OP("asinh");

TVM_TIR_REGISTER_PURE_UNARY_OP("atanh");

TVM_TIR_REGISTER_PURE_UNARY_OP("clz");

// binary intrinsics
TVM_TIR_REGISTER_PURE_BINARY_OP("atan2");

TVM_TIR_REGISTER_PURE_BINARY_OP("nextafter");

TVM_TIR_REGISTER_PURE_BINARY_OP("hypot");

TVM_TIR_REGISTER_PURE_BINARY_OP("copysign");

TVM_TIR_REGISTER_PURE_BINARY_OP("ldexp");

TVM_TIR_REGISTER_OP("TVMBackendAllocWorkspace")
    .set_num_inputs(5)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendAllocWorkspace")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

TVM_TIR_REGISTER_OP("TVMBackendFreeWorkspace")
    .set_num_inputs(3)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendFreeWorkspace")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

// expose basic functions to node namespace
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("node._const",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    if (auto opt = args[0].try_cast<int64_t>()) {
                      *ret = tirx::MakeConst(args[1].cast<PrimType>(), *opt, args[2].cast<Span>());
                    } else if (auto opt = args[0].try_cast<double>()) {
                      *ret = tirx::MakeConst(args[1].cast<PrimType>(), *opt, args[2].cast<Span>());
                    } else {
                      TVM_FFI_THROW(InternalError)
                          << "First argument to tvm.tirx.const must be int, float, or bool, "
                          << "but instead received argument with type code "
                          << args[0].GetTypeKey();
                    }
                  })
      .def("node.LargeUIntImm", LargeUIntImm)
      .def("tirx.min_value", static_cast<PrimExpr (*)(PrimType, Span)>(&min_value))
      .def("tirx.max_value", static_cast<PrimExpr (*)(PrimType, Span)>(&max_value))
      .def("tirx.infinity", static_cast<PrimExpr (*)(PrimType, Span)>(&infinity))
      .def("tirx.abs", tvm::abs)
      .def("tirx.likely", tvm::likely)
      .def("tirx.isnan", tvm::isnan)
      .def("tirx.isfinite", tvm::isfinite)
      .def("tirx.isinf", tvm::isinf)
      .def("tirx.floor", tvm::floor)
      .def("tirx.ceil", tvm::ceil)
      .def("tirx.round", tvm::round)
      .def("tirx.nearbyint", tvm::nearbyint)
      .def("tirx.trunc", tvm::trunc)
      .def("tirx._cast",
           [](PrimType dtype, PrimExpr value, Span span) { return tvm::cast(dtype, value, span); })
      .def("tirx.reinterpret",
           [](Type dtype, Expr value, Span span) { return tvm::reinterpret(dtype, value, span); });
}

// operator overloading, smarter than make
#define DEF_MAKE_BINARY_OP(Node, Func) \
  def("tirx." #Node, [](PrimExpr a, PrimExpr b, Span span) { return (Func(a, b, span)); })

#define DEF_MAKE_BIT_OP(Node, Func)                                                            \
  def_packed("tirx." #Node, [](ffi::PackedArgs args, ffi::Any* ret) {                          \
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
      .def("tirx._OpIfThenElse",
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
      .DEF_MAKE_BINARY_OP(_OpLogAddExp, logaddexp)
      .DEF_MAKE_BINARY_OP(_OpFloorMod, floormod)
      .DEF_MAKE_BINARY_OP(_OpTruncDiv, truncdiv)
      .DEF_MAKE_BINARY_OP(_OpTruncMod, truncmod)
      .DEF_MAKE_BINARY_OP(_OpCeilDiv, ceildiv)
      .DEF_MAKE_BINARY_OP(_OpPow, pow)
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

PrimExpr fast_erf_float_expr(PrimExpr arg, int bits) {
  PrimType fp_ty = PrimType::Float(bits);
  auto plus_4 = FloatImm(fp_ty, 4.f);
  auto minus_4 = FloatImm(fp_ty, -4.f);

  // The monomial coefficients of the numerator polynomial (odd).
  auto alpha_1 = FloatImm(fp_ty, -1.60960333262415e-02f);
  auto alpha_3 = FloatImm(fp_ty, -2.95459980854025e-03f);
  auto alpha_5 = FloatImm(fp_ty, -7.34990630326855e-04f);
  auto alpha_7 = FloatImm(fp_ty, -5.69250639462346e-05f);
  auto alpha_9 = FloatImm(fp_ty, -2.10102402082508e-06f);
  auto alpha_11 = FloatImm(fp_ty, 2.77068142495902e-08f);
  auto alpha_13 = FloatImm(fp_ty, -2.72614225801306e-10f);

  // The monomial coefficients of the denominator polynomial (even).
  auto beta_0 = FloatImm(fp_ty, -1.42647390514189e-02f);
  auto beta_2 = FloatImm(fp_ty, -7.37332916720468e-03f);
  auto beta_4 = FloatImm(fp_ty, -1.68282697438203e-03f);
  auto beta_6 = FloatImm(fp_ty, -2.13374055278905e-04f);
  auto beta_8 = FloatImm(fp_ty, -1.45660718464996e-05f);

  // clamp x
  auto x = tvm::max(tvm::min(arg, plus_4), minus_4);
  auto x2 = x * x;

  // Evaluate the numerator polynomial p.
  auto p = x2 * alpha_13 + alpha_11;
  p = x2 * p + alpha_9;
  p = x2 * p + alpha_7;
  p = x2 * p + alpha_5;
  p = x2 * p + alpha_3;
  p = x2 * p + alpha_1;
  p = x * p;

  // Evaluate the denominator polynomial p.
  auto q = x2 * beta_8 + beta_6;
  q = x2 * q + beta_4;
  q = x2 * q + beta_2;
  q = x2 * q + beta_0;

  return p / q;
}

// Helper function to safely extract boolean from PackedArgs
bool ExtractBool(const ffi::PackedArgs& args, int index) {
  try {
    return args[index].cast<bool>();
  } catch (...) {
    // Handle IntImm case (from TIR parsing)
    PrimExpr expr = args[index].cast<PrimExpr>();
    if (auto int_imm = expr.as<IntImmNode>()) {
      return int_imm->value != 0;
    }
    LOG(FATAL) << "Cannot extract bool from argument at index " << index;
    return false;
  }
}

// Helper function to safely extract int from PackedArgs
int ExtractInt(const ffi::PackedArgs& args, int index) {
  try {
    return args[index].cast<int>();
  } catch (...) {
    // Handle IntImm case (from TIR parsing)
    PrimExpr expr = args[index].cast<PrimExpr>();
    if (auto int_imm = expr.as<IntImmNode>()) {
      return static_cast<int>(int_imm->value);
    }
    LOG(FATAL) << "Cannot extract int from argument at index " << index;
    return 0;
  }
}

PrimExpr PrintOpPacked(Var data, DLDataType dtype, bool is_string, bool is_scalar, int dim_num,
                       ffi::Array<PrimExpr> shape) {
  PrimType value_ty(dtype);
  PrimType u32_ty = PrimType::UInt(32);
  ffi::Array<Expr> args;
  args.push_back(data);
  args.push_back(tirx::StringImm(ffi::DLDataTypeToString(dtype)));
  args.push_back(IntImm::Bool(is_string));
  args.push_back(IntImm::Bool(is_scalar));
  args.push_back(IntImm(u32_ty, dim_num));
  for (const auto& dim : shape) {
    args.push_back(dim);
  }
  return Call(value_ty, tirx::builtin::print_buffer(), args).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tirx.print_buffer", [](ffi::PackedArgs args, ffi::Any* ret) {
    // Expected arguments:
    // args[0]: buffer_var (Var)
    // args[1]: dtype (DLDataType)
    // args[2]: is_string (bool or IntImm)
    // args[3]: is_scalar (bool or IntImm)
    // args[4]: dim_num (int or IntImm)
    // args[5...]: shape dimensions (PrimExpr)

    TVM_FFI_ICHECK_GE(args.size(), 5) << "print_buffer expects at least 5 arguments";

    Var buffer_var = args[0].cast<Var>();
    DLDataType dtype = args[1].cast<DLDataType>();
    bool is_string = ExtractBool(args, 2);
    bool is_scalar = ExtractBool(args, 3);
    int dim_num = ExtractInt(args, 4);

    ffi::Array<PrimExpr> shape;
    for (int i = 5; i < args.size(); ++i) {
      shape.push_back(args[i].cast<PrimExpr>());
    }

    *ret = PrintOpPacked(buffer_var, dtype, is_string, is_scalar, dim_num, shape);
  });
}

}  // namespace tvm
