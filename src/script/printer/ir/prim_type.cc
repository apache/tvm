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

#include <tvm/runtime/memory_scope.h>
#include <tvm/script/printer/doc_translator.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> MemoryScopeDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                               const ffi::Object*) {
  return NamespaceDoc("tirx")
      ->Attr("MemoryScope")
      ->Call({LiteralDoc::Str(input.cast<runtime::MemoryScope>().name(), std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<runtime::MemoryScopeObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&MemoryScopeDocTranslate>());
}

bool HasPrimTypeShorthand(const PrimType& type) {
  bool supported_element =
      (type.MatchesCode(kDLInt, kDLUInt) &&
       (type.bits() == 8 || type.bits() == 16 || type.bits() == 32 || type.bits() == 64)) ||
      (type.MatchesCode(kDLFloat) &&
       (type.bits() == 16 || type.bits() == 32 || type.bits() == 64)) ||
      type.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3, kDLFloat8_e4m3b11fnuz, kDLFloat8_e4m3fn,
                       kDLFloat8_e4m3fnuz, kDLFloat8_e5m2, kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu,
                       kDLFloat6_e2m3fn, kDLFloat6_e3m2fn, kDLFloat4_e2m1fn);
  bool supported_lanes = type->dtype.lanes == 1 || type->dtype.lanes == 2 ||
                         type->dtype.lanes == 4 || type->dtype.lanes == 8 ||
                         type->dtype.lanes == 16 || type->dtype.lanes == 32 ||
                         type->dtype.lanes == 64;
  bool supported_scalar =
      type.IsVoid() || (type.IsScalar() && (type.MatchesElementType(kDLBool, 8) ||
                                            type.MatchesElementType(kDLBfloat, 16)));
  return (supported_element && supported_lanes) || supported_scalar;
}

ffi::Optional<ExprDoc> IntImmDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object*) {
  const auto* imm =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IntImmNode>(input);
  DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
  ExprDoc value = LiteralDoc::Int(ffi::GetRef<IntImm>(imm), std::nullopt);
  DLDataType default_dtype =
      ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.int_dtype", "int32"));
  if (dtype == default_dtype && !d->GetOrCreateExtraState<bool>("ir.explicit_value_type"))
    return value;
  if (!HasPrimTypeShorthand(imm->ty.as_or_throw<PrimType>()) ||
      imm->value < std::numeric_limits<int64_t>::min() ||
      imm->value > std::numeric_limits<int64_t>::max()) {
    return NamespaceDoc("tirx")->Attr("IntImm")->Call(
        {LiteralDoc::DataType(dtype, std::nullopt), value});
  }
  return NamespaceDoc("tirx")->Attr(ffi::DLDataTypeToString(dtype))->Call({value});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<IntImmNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&IntImmDocTranslate>());
}

ffi::Optional<ExprDoc> FloatImmDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* imm =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FloatImmNode>(input);
  DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
  DLDataType default_dtype =
      ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.float_dtype", "void"));
  ExprDoc value = LiteralDoc::Float(imm->value, std::nullopt);
  if (dtype == default_dtype && !d->GetOrCreateExtraState<bool>("ir.explicit_value_type"))
    return value;
  if (!HasPrimTypeShorthand(imm->ty.as_or_throw<PrimType>())) {
    if (!std::isfinite(imm->value)) {
      // Nonfinite literals render as strings; FloatImm requires a numeric value.
      value = NamespaceDoc("tirx")->Attr("float64")->Call({value})->Attr("value");
    }
    return NamespaceDoc("tirx")
        ->Attr("FloatImm")
        ->Call({LiteralDoc::DataType(dtype, std::nullopt), value});
  }
  return NamespaceDoc("tirx")->Attr(ffi::DLDataTypeToString(dtype))->Call({value});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<FloatImmNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&FloatImmDocTranslate>());
}

ffi::Optional<ExprDoc> PrimTypeDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PrimTypeNode>(input);
  PrimType type = ffi::GetRef<PrimType>(ty);
  // Other widths and fixed or scalable vectors have no exported TIRx shorthand.
  if (HasPrimTypeShorthand(type)) {
    return NamespaceDoc("tirx")->Attr(type.IsVoid() ? "void" : ffi::DLDataTypeToString(ty->dtype));
  }
  return NamespaceDoc("ir")
      ->Attr("PrimType")
      ->Call({LiteralDoc::DataType(ty->dtype, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<PrimTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&PrimTypeDocTranslate>());
}

ffi::Optional<ExprDoc> PtrTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PtrTypeNode>(input);
  return d->WithExtraStateScope<bool>("ir.explicit_value_type", true, [&]() -> ExprDoc {
    ffi::Array<ExprDoc> args{d->Translate(ty->element_type).value()};
    if (ty->scope != runtime::MemoryScope::Global()) {
      args.push_back(LiteralDoc::Str(ty->scope.name(), std::nullopt));
    }
    return NamespaceDoc("ir")->Attr("Ptr")->Call(args);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<PtrTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&PtrTypeDocTranslate>());
}

// The short builder name is already the TensorRegion expression constructor.
// Preserve that export and keep the explicit Type spelling for this one collision.
ffi::Optional<ExprDoc> TensorRegionTypeDocTranslate(DocTranslatorObj*, ffi::AnyView,
                                                    const ffi::Object*) {
  return NamespaceDoc("ir")->Attr("TensorRegionType")->Call({});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TensorRegionTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&TensorRegionTypeDocTranslate>());
}

ffi::Optional<ExprDoc> TupleTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeNode>(input);
  ExprDoc callee = NamespaceDoc("ir")->Attr("Tuple");
  ffi::Array<ExprDoc> fields;
  for (const Type& field : ty->fields) fields.push_back(d->Translate(field).value());
  return callee->Call(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&TupleTypeDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
