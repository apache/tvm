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
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/type.h>
#include <tvm/tirx/var.h>

#include <cmath>
// Shared primitive type matching.
#include "../../ir/prim/op_utils.h"

namespace tvm::prim {

using namespace prim::detail;
using tirx::TGlobalSymbol;
using tirx::TIRxOpCategory;
using tirx::TVectorizable;

template <size_t N>
static Type InferTypeReturnArgType(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

// macro to register an unary op

// macro to register an binary op

Type GetType(const PrimExpr& expr) {
  // TODO(tqchen): add recursive type inference for Call here
  // once we introduced the corresponding fields to the IR.
  if (auto* ptr = expr.as<tirx::VarNode>()) {
    // If Var has a more refined type annotation,
    // return the type anotation
    if (!ptr->ty.as<MissingType>().has_value()) {
      return ptr->ty;
    }
  }

  static const Op type_annotation_op = Op::Get("tirx.type_annotation");
  if (auto* access = expr.as<CallNode>()) {
    if (access->op.same_as(tirx::tvm_access_ptr_op())) {
      TVM_FFI_ICHECK(access->args.size())
          << "Builtin tvm_access_ptr() may not have empty arguments";
      auto type_annotation = access->args[0].as_or_throw<Call>();
      TVM_FFI_ICHECK(type_annotation->op.same_as(type_annotation_op))
          << "Expected the first argument of builtin tvm_access_ptr() "
          << "to be a type annotation, but found " << type_annotation->op;
      return PointerType(type_annotation->ty.as_or_throw<PrimType>());
    }
    if (access->op.same_as(tirx::ptr_byte_offset_op())) {
      TVM_FFI_ICHECK_EQ(access->args.size(), 3U);
      auto type_annotation = access->args[2].as_or_throw<Call>();
      TVM_FFI_ICHECK(type_annotation->op.same_as(type_annotation_op))
          << "Expected the third argument of builtin ptr_byte_offset() "
          << "to be a type annotation, but found " << type_annotation->op;
      return PointerType(type_annotation->ty.as_or_throw<PrimType>());
    }
  }

  if (auto* address_of = expr.as<CallNode>()) {
    if (address_of->op.same_as(tirx::address_of_op())) {
      TVM_FFI_ICHECK_EQ(address_of->args.size(), 1)
          << "Builtin address_of() expects a single argument, but received arguments "
          << address_of->args;
      auto* address = address_of->args[0].as<TensorLoadNode>();
      if (address) {
        return PointerType(address->ty.as_or_throw<PrimType>());
      }

      if (auto var = address_of->args[0].as<Var>()) {
        if (auto* ptr = var.value()->ty.as<PointerTypeNode>()) {
          if (ptr->element_type.as<tirx::TensorMapTypeNode>()) {
            return PrimType::UInt(64);
          }
        }
        return PointerType(var.value()->ty.as_or_throw<PrimType>());
      }

      TVM_FFI_ICHECK(false)
          << "Builtin address_of() expects the argument to be a TensorLoad or Var, but "
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

// Q-multiplication
PrimExpr q_multiply_shift(PrimExpr x, PrimExpr y, PrimExpr q, PrimExpr s, Span span) {
  return Call(PrimType::Int(32, x.ty().lanes()), tirx::q_multiply_shift_op(), {x, y, q, s}, {}, {},
              span)
      .as_or_throw<PrimExpr>();
}

PrimExpr thread_return(Span span) {
  return Call(PrimType::Void(), tirx::thread_return_op(), {}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.thread_return", thread_return);
}

PrimExpr logaddexp(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << b;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr exp_sum = add(exp(a), exp(b));
  PrimExpr log_exp_sum = log(exp_sum);
  return log_exp_sum;
}

// infinity
PrimExpr infinity(PrimType value_ty, Span span) {
  using namespace tirx;
  PrimType dtype = value_ty;
  TVM_FFI_ICHECK_EQ(dtype.lanes(), 1);
  if (dtype.MatchesCode(DLDataTypeCode::kDLFloat)) {
    if (dtype.bits() == 64) {
      return FloatImm(value_ty, std::numeric_limits<double>::infinity(), span);
    } else if (dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(value_ty, std::numeric_limits<float>::infinity(), span);
    }
  }
  TVM_FFI_THROW(InternalError) << "Cannot decide infinity for type " << dtype;
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
  return Call(std::move(t), tirx::reinterpret_op(), {value}, {}, {}, span).as_or_throw<PrimExpr>();
}

Expr reinterpret(Type target_ty, Expr value, Span span) {
  if (value.as<StringImmNode>()) {
    TVM_FFI_CHECK(target_ty.as<PointerTypeNode>(), TypeError)
        << "String reinterpret requires a pointer target, but got " << target_ty;
    return Call(std::move(target_ty), tirx::reinterpret_op(), {std::move(value)}, {}, {},
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
  return Call(std::move(target_ty), tirx::reinterpret_op(), {std::move(value)}, {}, {},
              std::move(span));
}

PrimExpr reinterpret(DLDataType t, PrimExpr value, Span span) {
  return reinterpret(PrimType(t), std::move(value), std::move(span));
}

// operator+
// pow
PrimExpr pow(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
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

  static const Op pow_op = Op::Get("tirx.pow");
  return Call(x.ty(), pow_op, {x, y}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.pow")
      .signature(sig::arg("x", "The input value."), sig::arg("y", "The second input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.pow"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);
}

// abs
PrimExpr abs(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt)) {
    return prim::IntegerAbs(x, span);
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat)) {
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return FloatImm(x.ty(), std::fabs(fx->value), fx->span);
    }
    static const Op fabs_op = Op::Get("tirx.fabs");
    return Call(x.ty(), fabs_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLUInt)) {
    return x;
  } else {
    TVM_FFI_THROW(InternalError) << "Data type " << x.ty()
                                 << " not supported for absolute op. Skipping absolute op...";
    return x;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.fabs")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fabs"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);
}

// isnan
PrimExpr isnan(PrimExpr x, Span span) {
  PrimType t = PrimType::Bool(x.ty().lanes());
  PrimType bool_ty(t);
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return MakeConst(t, false);
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLFloat)) {
    const FloatImmNode* fx = x.as<FloatImmNode>();
    if (fx) {
      return MakeConst(t, std::isnan(fx->value), fx->span);
    }
    if (x.ty().bits() == 16) {
      static const Op isnan_op = Op::Get("tirx.isnan");
      PrimType f32_ty = PrimType::Float(32, t.lanes());
      return Call(bool_ty, isnan_op, {cast(f32_ty, std::move(x), span)}, {}, {}, span)
          .as_or_throw<PrimExpr>();
    } else {
      static const Op isnan_op = Op::Get("tirx.isnan");
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
  } else if (x.ty().MatchesCode(DLDataTypeCode::kDLFloat)) {
    PrimExpr infX = infinity(x.ty(), span);
    return abs(x, span) == infX && !isnan(x, span);
  } else {
    TVM_FFI_THROW(InternalError) << "Data type " << x.ty()
                                 << " not supported for finiteness ops. Skipping it...";
  }
}

// isfinite
PrimExpr isfinite(PrimExpr x, Span span) { return !isinf(x, span) && !isnan(x, span); }

// fmod
PrimExpr fmod(PrimExpr x, PrimExpr y, Span span) {
  BinaryOpMatchTypes(x, y, span);
  TVM_FFI_ICHECK(x.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << "fmod only applies to float";
  static const Op fmod_op = Op::Get("tirx.fmod");
  return Call(x.ty(), fmod_op, {x, y}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.fmod")
      .signature(sig::arg("x", "The input value."), sig::arg("y", "The second input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.fmod"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

// floor
PrimExpr floor(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::floor(fx->value), fx->span);
  static const Op floor_op = Op::Get("tirx.floor");
  return Call(x.ty(), floor_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.floor")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.floor"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);
}

// round
PrimExpr round(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::nearbyint(fx->value), fx->span);
  static const Op round_op = Op::Get("tirx.round");
  return Call(x.ty(), round_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.round")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.round"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);
}

// nearbyint
PrimExpr nearbyint(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) return FloatImm(x.ty(), std::nearbyint(fx->value), fx->span);
  static const Op nearbyint_op = Op::Get("tirx.nearbyint");
  return Call(x.ty(), nearbyint_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.nearbyint")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.nearbyint"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

// trunc
PrimExpr trunc(PrimExpr x, Span span) {
  if (x.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                         DLDataTypeCode::kDLBool)) {
    return x;
  }
  const FloatImmNode* fx = x.as<FloatImmNode>();
  if (fx) {
    return FloatImm(x.ty(), (fx->value < 0 ? std::ceil(fx->value) : std::floor(fx->value)),
                    fx->span);
  }
  static const Op trunc_op = Op::Get("tirx.trunc");
  return Call(x.ty(), trunc_op, {x}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.trunc")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.trunc"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  // unary op registration.

  OpDef("tirx.exp")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.exp2")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.exp10")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.exp10"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.erf")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.erf"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.tanh")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tanh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.sigmoid")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sigmoid"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.sqrt")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.rsqrt")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.rsqrt"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.log")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.log1p")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log1p"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.log10")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.log10"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.tan")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.tan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.cos")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cos"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.cosh")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.cosh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.sin")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.sinh")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.sinh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TVectorizable>("TVectorizable", true);

  OpDef("tirx.asin")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.asin"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.acos")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.acos"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.atan")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atan"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.acosh")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.acosh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.asinh")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.asinh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.atanh")
      .signature(sig::arg("x", "The input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atanh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  // binary intrinsics

  OpDef("tirx.atan2")
      .signature(sig::arg("x1", "The first input value."),
                 sig::arg("x2", "The second input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.atan2"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.nextafter")
      .signature(sig::arg("x1", "The first input value."),
                 sig::arg("x2", "The second input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.nextafter"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.hypot")
      .signature(sig::arg("x1", "The first input value."),
                 sig::arg("x2", "The second input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.hypot"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.copysign")
      .signature(sig::arg("x1", "The first input value."),
                 sig::arg("x2", "The second input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.copysign"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.ldexp")
      .signature(sig::arg("x1", "The first input value."),
                 sig::arg("x2", "The second input value."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeReturnArgType<0>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ldexp"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("tirx.TVMBackendAllocWorkspace")
      .set_attr<TFixedReturnType>("TFixedReturnType", PointerType(PrimType::Void()))
      .signature(sig::arg<IntExpr>("device_type", "The device type."),
                 sig::arg<IntExpr>("device_id", "The device index."),
                 sig::arg<IntExpr>("nbytes", "The number of bytes."),
                 sig::arg<IntExpr>("dtype_code_hint", "The data type code hint."),
                 sig::arg<IntExpr>("dtype_bits_hint", "The data type bit-width hint."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.TVMBackendAllocWorkspace"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendAllocWorkspace")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.TVMBackendFreeWorkspace")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .signature(sig::arg<IntExpr>("device_type", "The device type."),
                 sig::arg<IntExpr>("device_id", "The device index."),
                 sig::arg("ptr", "The pointer."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.TVMBackendFreeWorkspace"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "TVMBackendFreeWorkspace")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  // expose basic functions to node namespace

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.infinity", static_cast<PrimExpr (*)(PrimType, Span)>(&infinity))
      .def("tirx.abs", prim::abs)
      .def("tirx.isnan", prim::isnan)
      .def("tirx.isfinite", prim::isfinite)
      .def("tirx.isinf", prim::isinf)
      .def("tirx.floor", prim::floor)
      .def("tirx.round", prim::round)
      .def("tirx.nearbyint", prim::nearbyint)
      .def("tirx.trunc", prim::trunc)
      .def("tirx.reinterpret",
           [](Type dtype, Expr value, Span span) { return prim::reinterpret(dtype, value, span); });

  tvm::ffi::reflection::GlobalDef()
      .def("tirx._OpPow", [](PrimExpr a, PrimExpr b, Span span) { return pow(a, b, span); })
      .def("tirx._OpLogAddExp",
           [](PrimExpr a, PrimExpr b, Span span) { return logaddexp(a, b, span); });
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
      auto value = int_imm->value.as<int>();
      TVM_FFI_CHECK(value.has_value(), OverflowError) << "Integer argument does not fit int";
      return *value;
    }
    LOG(FATAL) << "Cannot extract int from argument at index " << index;
    return 0;
  }
}

PrimExpr PrintOpPacked(Expr data, DLDataType dtype, bool is_string, bool is_scalar, int dim_num,
                       ffi::Array<PrimExpr> shape) {
  PrimType value_ty(dtype);
  PrimType u32_ty = PrimType::UInt(32);
  ffi::Array<Expr> args;
  args.push_back(data);
  args.push_back(StringImm(ffi::DLDataTypeToString(dtype)));
  args.push_back(IntImm::Bool(is_string));
  args.push_back(IntImm::Bool(is_scalar));
  args.push_back(IntImm(u32_ty, dim_num));
  for (const auto& dim : shape) {
    args.push_back(dim);
  }
  return Call(value_ty, tirx::print_buffer_op(), args).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("tirx.print_buffer", [](ffi::PackedArgs args, ffi::Any* ret) {
    // Expected arguments:
    // args[0]: buffer data expression
    // args[1]: dtype (DLDataType)
    // args[2]: is_string (bool or IntImm)
    // args[3]: is_scalar (bool or IntImm)
    // args[4]: dim_num (int or IntImm)
    // args[5...]: shape dimensions (PrimExpr)

    TVM_FFI_ICHECK_GE(args.size(), 5) << "print_buffer expects at least 5 arguments";

    Expr buffer_data = args[0].cast<Expr>();
    DLDataType dtype = args[1].cast<DLDataType>();
    bool is_string = ExtractBool(args, 2);
    bool is_scalar = ExtractBool(args, 3);
    int dim_num = ExtractInt(args, 4);

    ffi::Array<PrimExpr> shape;
    for (int i = 5; i < args.size(); ++i) {
      shape.push_back(args[i].cast<PrimExpr>());
    }

    *ret = PrintOpPacked(buffer_data, dtype, is_string, is_scalar, dim_num, shape);
  });
}

}  // namespace tvm::prim
