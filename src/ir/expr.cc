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
 * \file src/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/type.h>
#include <tvm/te/tensor.h>
#include <tvm/tirx/expr.h>

#include <cmath>

#include "../support/limits.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  ExprNode::RegisterReflection();
  BaseFuncNode::RegisterReflection();
  GlobalVarNode::RegisterReflection();
  CallNode::RegisterReflection();
  IntImmNode::RegisterReflection();
  FloatImmNode::RegisterReflection();
  RangeNode::RegisterReflection();
}

PrimExpr::PrimExpr(Call call) : PrimExpr(std::move(call).as_or_throw<PrimExpr>()) {}

PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm::Int32(value)) {}

PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(PrimType::Float(32), value)) {}

PrimExpr PrimExpr::ConvertFallbackValue(ffi::String value) { return tirx::StringImm(value); }

namespace ffi {

PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(StrictBool value) {
  return IntImm::Bool(value);
}

PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(int64_t value) {
  return TypeTraits<IntImm>::ConvertFallbackValue(value);
}

PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(double value) {
  return TypeTraits<FloatImm>::ConvertFallbackValue(value);
}

}  // namespace ffi

IntImm::IntImm(PrimType value_ty, int64_t value, Span span) {
  DLDataType runtime_dtype = value_ty->dtype;
  DLDataTypeCode code = value_ty.code();
  int32_t bits = value_ty.bits();
  TVM_FFI_CHECK(!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector(), ValueError)
      << "IntImm can only take scalar, but " << runtime_dtype << " was supplied.";
  TVM_FFI_CHECK(value_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                     DLDataTypeCode::kDLBool),
                ValueError)
      << "IntImm supports only int or uint or bool type, but " << runtime_dtype << " was supplied.";
  if (code == DLDataTypeCode::kDLUInt) {
    TVM_FFI_CHECK_GE(value, 0U, ValueError)
        << "Literal value " << value << " is negative for unsigned integer type " << runtime_dtype;
    if (bits < 64) {
      TVM_FFI_CHECK_LT(value, 1LL << bits, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    }
  } else if (bits == 1 || code == DLDataTypeCode::kDLBool) {
    // int(1)
    TVM_FFI_CHECK(value == 0 || value == 1, ValueError)
        << value << " exceeds range of " << runtime_dtype;
  } else if (bits < 64) {
    TVM_FFI_CHECK_GE(value, -(1LL << (bits - 1)), ValueError)
        << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
    TVM_FFI_CHECK_LT(value, 1LL << (bits - 1), ValueError)
        << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
  }
  ffi::ObjectPtr<IntImmNode> node = ffi::make_object<IntImmNode>();
  node->ExprNode::ty = std::move(value_ty);
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.IntImm", [](DLDataType dtype, int64_t value, Span span) {
    return IntImm(PrimType(dtype), value, span);
  });
}

FloatImm::FloatImm(PrimType value_ty, double value, Span span) {
  DLDataType runtime_dtype = value_ty->dtype;
  DLDataTypeCode code = value_ty.code();
  int32_t bits = value_ty.bits();
  TVM_FFI_CHECK(!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector(), ValueError)
      << "FloatImm can only take scalar.";

  TVM_FFI_CHECK(
      value_ty.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLFloat8_e3m4,
                           DLDataTypeCode::kDLFloat8_e4m3, DLDataTypeCode::kDLFloat8_e4m3b11fnuz,
                           DLDataTypeCode::kDLFloat8_e4m3fn, DLDataTypeCode::kDLFloat8_e4m3fnuz,
                           DLDataTypeCode::kDLFloat8_e5m2, DLDataTypeCode::kDLFloat8_e5m2fnuz,
                           DLDataTypeCode::kDLFloat8_e8m0fnu, DLDataTypeCode::kDLFloat6_e2m3fn,
                           DLDataTypeCode::kDLFloat6_e3m2fn) ||
          value_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16) ||
          value_ty.MatchesElementType(DLDataTypeCode::kDLFloat4_e2m1fn, 4) ||
          static_cast<int>(code) >= static_cast<int>(ffi::DLExtDataTypeCode::kDLExtCustomBegin),
      ValueError)
      << "FloatImm supports only float, but " << runtime_dtype << " was supplied.";

  // check range for float32 and float16 since they have specified range.
  if (!std::isinf(value) && !std::isnan(value)) {
    if (bits == 32) {
      TVM_FFI_CHECK_GE(value, std::numeric_limits<float>::lowest(), ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, std::numeric_limits<float>::max(), ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (value_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16)) {
      TVM_FFI_CHECK_GE(value, -support::kMaxFloat16, ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, support::kMaxFloat16, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (value_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
      TVM_FFI_CHECK_GE(value, -support::kMaxBFloat16, ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, support::kMaxBFloat16, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (value_ty.MatchesCode(
                   DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3,
                   DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn,
                   DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2,
                   DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu)) {
      double bound = 0.0;
      bool nonneg = false;

      switch (code) {
        case DLDataTypeCode::kDLFloat8_e3m4:
          bound = support::kMaxE3M4;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3:
          bound = support::kMaxE4M3;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3b11fnuz:
          bound = support::kMaxE4M3B11FNUZ;
          nonneg = true;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3fn:
          bound = support::kMaxE4M3FN;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3fnuz:
          bound = support::kMaxE4M3FNUZ;
          nonneg = true;
          break;
        case DLDataTypeCode::kDLFloat8_e5m2:
          bound = support::kMaxE5M2;
          break;
        case DLDataTypeCode::kDLFloat8_e5m2fnuz:
          bound = support::kMaxE5M2FNUZ;
          nonneg = true;
          break;
        case DLDataTypeCode::kDLFloat8_e8m0fnu:
          bound = support::kMaxE8M0FNU;
          nonneg = true;
          break;
        default:
          TVM_FFI_THROW(InternalError) << "Unhandled float8 type: " << runtime_dtype;
      }

      if (nonneg) {
        TVM_FFI_CHECK_GE(value, 0, ValueError)
            << "Literal value " << value << " below zero for unsigned " << runtime_dtype;
      } else {
        TVM_FFI_CHECK_GE(value, -bound, ValueError)
            << "Literal value " << value << " below minimum of " << runtime_dtype;
      }
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;

    } else if (value_ty.MatchesCode(DLDataTypeCode::kDLFloat6_e2m3fn,
                                    DLDataTypeCode::kDLFloat6_e3m2fn)) {
      double bound =
          (code == DLDataTypeCode::kDLFloat6_e2m3fn) ? support::kMaxE2M3FN : support::kMaxE3M2FN;
      TVM_FFI_CHECK_GE(value, -bound, ValueError)
          << "Literal value " << value << " below minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;

    } else if (code == DLDataTypeCode::kDLFloat4_e2m1fn) {
      double bound = support::kMaxE2M1FN;
      TVM_FFI_CHECK_GE(value, -bound, ValueError)
          << "Literal value " << value << " below minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    }
  }
  ffi::ObjectPtr<FloatImmNode> node = ffi::make_object<FloatImmNode>();
  node->ExprNode::ty = std::move(value_ty);
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.FloatImm", [](DLDataType dtype, double value, Span span) {
    return FloatImm(PrimType(dtype), value, span);
  });
}

Range::Range(PrimExpr begin, PrimExpr end, Span span)
    : Range(ffi::make_object<RangeNode>(begin, tirx::is_zero(begin) ? end : (end - begin), span)) {}

Range Range::FromMinExtent(PrimExpr min, PrimExpr extent, Span span) {
  return Range(ffi::make_object<RangeNode>(min, extent, span));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.Range_from_min_extent", Range::FromMinExtent)
      .def("ir.Range", [](PrimExpr begin, ffi::Optional<PrimExpr> end, Span span) -> Range {
        if (end.defined()) {
          return Range(begin, end.value(), span);
        } else {
          return Range(IntImm(begin.ty(), 0), begin, span);
        }
      });
}

GlobalVar::GlobalVar(ffi::String name_hint, Span span) {
  ffi::ObjectPtr<GlobalVarNode> n = ffi::make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  n->span = std::move(span);
  data_ = std::move(n);
}

Call::Call(Type ret_ty, Expr op, ffi::Array<Expr> args, Attrs attrs, ffi::Array<Type> ty_args,
           Span span) {
  TVM_FFI_CHECK(op.defined(), ValueError) << "Call expects a defined operator";

  ffi::ObjectPtr<CallNode> n = ffi::make_object<CallNode>();
  n->ExprNode::ty = std::move(ret_ty);
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->ty_args = std::move(ty_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.GlobalVar", [](ffi::String name) { return GlobalVar(name); })
      .def("ir.Call",
           [](Type ret_ty, Expr op, ffi::Array<Expr> args, Attrs attrs, ffi::Array<Type> ty_args,
              Span span) { return Call(ret_ty, op, args, attrs, ty_args, span); })
      .def("ir.DebugPrint", [](ffi::ObjectRef ref) {
        std::stringstream ss;
        ss << ref;
        return ss.str();
      });
  // Note: kRepr for GlobalVarNode is registered in script/printer/ir/ir.cc
  // via TVM_REGISTER_SCRIPT_AS_REPR(GlobalVarNode, ReprPrintIR).
}

}  // namespace tvm
