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
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op_attr_types.h>

#include <cmath>

#include "op_utils.h"

namespace tvm {
namespace prim {

using namespace prim::detail;

template <size_t N>
static Type InferTypeReturnArgType(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

Type InferTypeIsNaN(const CallNode* call) {
  TVM_FFI_CHECK_GE(call->args.size(), 1U, ValueError) << "prim.isnan expects one argument";
  PrimType input = call->args[0]->ty.as_or_throw<PrimType>();
  if (input.IsScalableVector()) {
    return PrimType::ScalableVector(DLDataTypeCode::kDLBool, PrimType::Bool().bits(),
                                    input.VScaleFactor());
  }
  return PrimType::Bool(input.lanes());
}

const Op& pow_op() {
  static const Op op = Op::Get("prim.pow");
  return op;
}

const Op& fabs_op() {
  static const Op op = Op::Get("prim.fabs");
  return op;
}

const Op& fmod_op() {
  static const Op op = Op::Get("prim.fmod");
  return op;
}

const Op& floor_op() {
  static const Op op = Op::Get("prim.floor");
  return op;
}

const Op& round_op() {
  static const Op op = Op::Get("prim.round");
  return op;
}

const Op& nearbyint_op() {
  static const Op op = Op::Get("prim.nearbyint");
  return op;
}

const Op& trunc_op() {
  static const Op op = Op::Get("prim.trunc");
  return op;
}

const Op& exp_op() {
  static const Op op = Op::Get("prim.exp");
  return op;
}

const Op& exp2_op() {
  static const Op op = Op::Get("prim.exp2");
  return op;
}

const Op& exp10_op() {
  static const Op op = Op::Get("prim.exp10");
  return op;
}

const Op& erf_op() {
  static const Op op = Op::Get("prim.erf");
  return op;
}

const Op& tanh_op() {
  static const Op op = Op::Get("prim.tanh");
  return op;
}

const Op& sigmoid_op() {
  static const Op op = Op::Get("prim.sigmoid");
  return op;
}

const Op& sqrt_op() {
  static const Op op = Op::Get("prim.sqrt");
  return op;
}

const Op& rsqrt_op() {
  static const Op op = Op::Get("prim.rsqrt");
  return op;
}

const Op& log_op() {
  static const Op op = Op::Get("prim.log");
  return op;
}

const Op& log1p_op() {
  static const Op op = Op::Get("prim.log1p");
  return op;
}

const Op& log10_op() {
  static const Op op = Op::Get("prim.log10");
  return op;
}

const Op& tan_op() {
  static const Op op = Op::Get("prim.tan");
  return op;
}

const Op& cos_op() {
  static const Op op = Op::Get("prim.cos");
  return op;
}

const Op& cosh_op() {
  static const Op op = Op::Get("prim.cosh");
  return op;
}

const Op& sin_op() {
  static const Op op = Op::Get("prim.sin");
  return op;
}

const Op& sinh_op() {
  static const Op op = Op::Get("prim.sinh");
  return op;
}

const Op& asin_op() {
  static const Op op = Op::Get("prim.asin");
  return op;
}

const Op& acos_op() {
  static const Op op = Op::Get("prim.acos");
  return op;
}

const Op& atan_op() {
  static const Op op = Op::Get("prim.atan");
  return op;
}

const Op& acosh_op() {
  static const Op op = Op::Get("prim.acosh");
  return op;
}

const Op& asinh_op() {
  static const Op op = Op::Get("prim.asinh");
  return op;
}

const Op& atanh_op() {
  static const Op op = Op::Get("prim.atanh");
  return op;
}

const Op& atan2_op() {
  static const Op op = Op::Get("prim.atan2");
  return op;
}

const Op& nextafter_op() {
  static const Op op = Op::Get("prim.nextafter");
  return op;
}

const Op& hypot_op() {
  static const Op op = Op::Get("prim.hypot");
  return op;
}

const Op& copysign_op() {
  static const Op op = Op::Get("prim.copysign");
  return op;
}

const Op& ldexp_op() {
  static const Op op = Op::Get("prim.ldexp");
  return op;
}

const Op& isnan_op() {
  static const Op op = Op::Get("prim.isnan");
  return op;
}

const Op& popcount_op() {
  static const Op op = Op::Get("prim.popcount");
  return op;
}

const Op& fma_op() {
  static const Op op = Op::Get("prim.fma");
  return op;
}

const Op& assume_op() {
  static const Op op = Op::Get("prim.assume");
  return op;
}

PrimExpr infinity(PrimType value_ty, ffi::Optional<Location> loc) {
  PrimType dtype = value_ty;
  TVM_FFI_ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.MatchesCode(DLDataTypeCode::kDLFloat)) {
    if (dtype.bits() == 64) {
      return FloatImm(value_ty, std::numeric_limits<double>::infinity(), loc);
    } else if (dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(value_ty, std::numeric_limits<float>::infinity(), loc);
    }
  }
  TVM_FFI_THROW(InternalError) << "Cannot decide infinity for type " << dtype;
}

PrimExpr pow(PrimExpr x, PrimExpr y, ffi::Optional<Location> loc) {
  BinaryOpMatchTypes(x, y, loc.value_or(UnknownLoc()));
  TVM_FFI_ICHECK(x.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << "power only applies to float";

  // If we detect pow(x, 3), suggest using x * x * x
  if (y.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    const IntImmNode* px = y.as<IntImmNode>();
    if (px) {
      if (px->value >= 3) {
        LOG(WARNING)
            << "Detected pow(x, y) where y >= 3, it is recommended to avoid this as it may lead to "
               "uninteded behaviors when x < 0. Perhaps with `x * x * x ...` or "
               "`pow(x, 2) * pow(x, 2) ...`.";
      }
    }
  } else if (y.ty().MatchesCode(DLDataTypeCode::kDLFloat)) {
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

  static const Op pow_op = Op::Get("prim.pow");
  return Call(x.ty(), pow_op, {x, y}, {}, {}, loc).as_or_throw<PrimExpr>();
}

PrimExpr abs(PrimExpr x, ffi::Optional<Location> loc) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    return prim::IntegerAbs(x, loc);
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat)) {
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return FloatImm(x.ty(), std::fabs(fx->value), fx->loc);
    }
    static const Op fabs_op = Op::Get("prim.fabs");
    return Call(x.ty(), fabs_op, {x}, {}, {}, loc).as_or_throw<PrimExpr>();
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLUInt)) {
    return x;
  } else {
    TVM_FFI_THROW(InternalError) << "Data type " << x.ty()
                                 << " not supported for absolute op. Skipping absolute op...";
    return x;
  }
}

PrimExpr isnan(PrimExpr x, ffi::Optional<Location> loc) {
  static const Op op = Op::Get("prim.isnan");
  return Call(std::nullopt, op, {std::move(x)}, {}, {}, loc).as_or_throw<PrimExpr>();
}

PrimExpr isinf(PrimExpr x, ffi::Optional<Location> loc) {
  PrimType t = PrimType::Bool(x.ty().lanes());
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return MakeConst(t, false, loc);
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLFloat)) {
    PrimExpr infX = infinity(x.ty(), loc);
    return abs(x, loc) == infX && !isnan(x, loc);
  } else {
    TVM_FFI_THROW(InternalError) << "Data type " << x.ty()
                                 << " not supported for finiteness ops. Skipping it...";
  }
}

PrimExpr isfinite(PrimExpr x, ffi::Optional<Location> loc) {
  return !isinf(x, loc) && !isnan(x, loc);
}

PrimExpr fmod(PrimExpr x, PrimExpr y, ffi::Optional<Location> loc) {
  BinaryOpMatchTypes(x, y, loc.value_or(UnknownLoc()));
  TVM_FFI_ICHECK(x.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << "fmod only applies to float";
  static const Op fmod_op = Op::Get("prim.fmod");
  return Call(x.ty(), fmod_op, {x, y}, {}, {}, loc).as_or_throw<PrimExpr>();
}

PrimExpr floor(PrimExpr x, ffi::Optional<Location> loc) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::floor(fx->value), fx->loc);
  static const Op floor_op = Op::Get("prim.floor");
  return Call(x.ty(), floor_op, {x}, {}, {}, loc).as_or_throw<PrimExpr>();
}

PrimExpr round(PrimExpr x, ffi::Optional<Location> loc) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::nearbyint(fx->value), fx->loc);
  static const Op round_op = Op::Get("prim.round");
  return Call(x.ty(), round_op, {x}, {}, {}, loc).as_or_throw<PrimExpr>();
}

PrimExpr nearbyint(PrimExpr x, ffi::Optional<Location> loc) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::nearbyint(fx->value), fx->loc);
  static const Op nearbyint_op = Op::Get("prim.nearbyint");
  return Call(x.ty(), nearbyint_op, {x}, {}, {}, loc).as_or_throw<PrimExpr>();
}

PrimExpr trunc(PrimExpr x, ffi::Optional<Location> loc) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) {
    return FloatImm(x.ty(), (fx->value < 0 ? std::ceil(fx->value) : std::floor(fx->value)),
                    fx->loc);
  }
  static const Op trunc_op = Op::Get("prim.trunc");
  return Call(x.ty(), trunc_op, {x}, {}, {}, loc).as_or_throw<PrimExpr>();
}

PrimExpr assume(PrimExpr condition, ffi::Optional<Location> loc) {
  return Call(PrimType::Bool(), assume_op(), {condition}, {}, {}, loc).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.pow")
      .signature(sig::arg<PrimExpr>("x", "The input value."),
                 sig::arg<PrimExpr>("y", "The second input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.fabs")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.fmod")
      .signature(sig::arg<PrimExpr>("x", "The input value."),
                 sig::arg<PrimExpr>("y", "The second input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.floor")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.round")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.nearbyint")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.trunc")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.exp")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.exp2")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.exp10")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.erf")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.tanh")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.sigmoid")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.sqrt")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.rsqrt")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.log")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.log1p")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.log10")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.tan")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.cos")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.cosh")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.sin")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.sinh")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.asin")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.acos")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.atan")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.acosh")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.asinh")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.atanh")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.atan2")
      .signature(sig::arg<PrimExpr>("x1", "The first input value."),
                 sig::arg<PrimExpr>("x2", "The second input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.nextafter")
      .signature(sig::arg<PrimExpr>("x1", "The first input value."),
                 sig::arg<PrimExpr>("x2", "The second input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.hypot")
      .signature(sig::arg<PrimExpr>("x1", "The first input value."),
                 sig::arg<PrimExpr>("x2", "The second input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.copysign")
      .signature(sig::arg<PrimExpr>("x1", "The first input value."),
                 sig::arg<PrimExpr>("x2", "The second input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.ldexp")
      .signature(sig::arg<PrimExpr>("x1", "The first input value."),
                 sig::arg<PrimExpr>("x2", "The second input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.isnan")
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<FInferType>(tvm::op_attr::kInferType, FInferType::FromNative<&InferTypeIsNaN>())
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.popcount")
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .signature(sig::arg<PrimExpr>("x", "The input value."))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.fma")
      .set_attr<FInferType>(tvm::op_attr::kInferType,
                            FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .signature(sig::arg<PrimExpr>("x", "The input value."),
                 sig::arg<PrimExpr>("y", "The second input value."),
                 sig::arg<PrimExpr>("z", "The third input value."))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.assume")
      .set_attr<TFixedReturnType>(tvm::op_attr::kFixedReturnType, PrimType::Bool())
      .signature(sig::arg<PrimExpr>("cond", "The condition."))
      .set_attr<TCallEffectKind>(tvm::op_attr::kCallEffectKind,
                                 static_cast<int64_t>(CallEffectKind::kEmbedInfo));

  ffi::reflection::GlobalDef()
      .def("prim.infinity", prim::infinity)
      .def("prim.abs", prim::abs)
      .def("prim.isnan", prim::isnan)
      .def("prim.isfinite", prim::isfinite)
      .def("prim.isinf", prim::isinf)
      .def("prim.floor", prim::floor)
      .def("prim.round", prim::round)
      .def("prim.nearbyint", prim::nearbyint)
      .def("prim.trunc", prim::trunc)
      .def("prim.assume", prim::assume)
      .def("prim._OpPow",
           [](PrimExpr a, PrimExpr b, ffi::Optional<Location> loc) { return pow(a, b, loc); });
}

}  // namespace prim
}  // namespace tvm
