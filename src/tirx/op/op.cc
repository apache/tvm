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

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op_attr_types.h>

#include "../../ir/prim/op_utils.h"

namespace tvm::tirx {

using namespace prim;
using namespace prim::detail;

PrimExpr thread_return(Span span) {
  return Call(PrimType::Void(), tirx::thread_return_op(), {}, {}, {}, span).as_or_throw<PrimExpr>();
}

PrimExpr logaddexp(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_ICHECK(a.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << a;
  TVM_FFI_ICHECK(b.ty().MatchesCode(DLDataTypeCode::kDLFloat)) << b;
  BinaryOpMatchTypes(a, b, span);
  PrimExpr exp_sum = add(exp(a), exp(b));
  PrimExpr log_exp_sum = log(exp_sum);
  return log_exp_sum;
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

TVM_FFI_STATIC_INIT_BLOCK() {
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
  ffi::reflection::GlobalDef()
      .def("tirx.thread_return", thread_return)
      .def("tirx.reinterpret",
           [](Type dtype, Expr value, Span span) { return reinterpret(dtype, value, span); })
      .def("tirx._OpLogAddExp",
           [](PrimExpr a, PrimExpr b, Span span) { return logaddexp(a, b, span); });
}

}  // namespace tvm::tirx
