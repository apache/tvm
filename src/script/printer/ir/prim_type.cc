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

#include <tvm/tirx/type.h>

#include <algorithm>
#include <functional>
#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> MissingTypeDocTranslate(DocTranslatorObj*, ffi::AnyView,
                                               const ffi::Object*) {
  return NamespaceDoc("ir")->Attr("MissingType")->Call({});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<MissingTypeNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&MissingTypeDocTranslate>());
}

ffi::Optional<ExprDoc> IntImmDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object*) {
  const auto* imm =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IntImmNode>(input);
  DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
  ExprDoc value = LiteralDoc::Int(ffi::GetRef<IntImm>(imm), std::nullopt);
  DLDataType default_dtype =
      ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.int_dtype", "int32"));
  if (dtype == default_dtype) return value;
  return NamespaceDoc("tirx")->Attr(ffi::DLDataTypeToString(dtype))->Call({value});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<IntImmNode>().attr(kDocTranslate,
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
  if (dtype == default_dtype) return value;
  return NamespaceDoc("tirx")->Attr(ffi::DLDataTypeToString(dtype))->Call({value});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<FloatImmNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&FloatImmDocTranslate>());
}

ffi::Optional<ExprDoc> PrimTypeDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PrimTypeNode>(input);
  PrimType type = ffi::GetRef<PrimType>(ty);
  bool supported_element =
      (type.MatchesCode(kDLInt, kDLUInt) &&
       (type.bits() == 8 || type.bits() == 16 || type.bits() == 32 || type.bits() == 64)) ||
      (type.MatchesCode(kDLFloat) &&
       (type.bits() == 16 || type.bits() == 32 || type.bits() == 64)) ||
      type.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3, kDLFloat8_e4m3b11fnuz, kDLFloat8_e4m3fn,
                       kDLFloat8_e4m3fnuz, kDLFloat8_e5m2, kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu,
                       kDLFloat6_e2m3fn, kDLFloat6_e3m2fn, kDLFloat4_e2m1fn);
  bool supported_lanes = ty->dtype.lanes == 1 || ty->dtype.lanes == 2 || ty->dtype.lanes == 4 ||
                         ty->dtype.lanes == 8 || ty->dtype.lanes == 16 || ty->dtype.lanes == 32 ||
                         ty->dtype.lanes == 64;
  bool supported_scalar = type.IsScalar() && (type.MatchesElementType(kDLBool, 8) ||
                                              type.MatchesElementType(kDLBfloat, 16));
  // Other widths and fixed or scalable vectors have no exported TIRx shorthand.
  if ((supported_element && supported_lanes) || supported_scalar) {
    return NamespaceDoc("tirx")->Attr(ffi::DLDataTypeToString(ty->dtype));
  }
  return NamespaceDoc("ir")
      ->Attr("PrimType")
      ->Call({LiteralDoc::DataType(ty->dtype, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<PrimTypeNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&PrimTypeDocTranslate>());
}

ffi::Optional<ExprDoc> StringTypeDocTranslate(DocTranslatorObj*, ffi::AnyView, const ffi::Object*) {
  return NamespaceDoc("ir")->Attr("StringType")->Call({});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<StringTypeNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&StringTypeDocTranslate>());
}

ffi::Optional<ExprDoc> TupleTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeNode>(input);
  if (ty->fields.empty()) return NamespaceDoc("relax")->Attr("Tuple");
  ffi::Array<ExprDoc> fields;
  for (const Type& field : ty->fields) fields.push_back(d->Translate(field).value());
  std::function<bool(const Type&)> is_primitive = [&](const Type& field) {
    if (field.as<PrimType>() || field.as<PointerType>()) return true;
    if (const auto* tuple = field.as<TupleTypeNode>()) {
      return std::all_of(tuple->fields.begin(), tuple->fields.end(), is_primitive);
    }
    return false;
  };
  bool primitive = std::all_of(ty->fields.begin(), ty->fields.end(), is_primitive);
  return (primitive ? NamespaceDoc("tirx")->Attr("Tuple") : NamespaceDoc("relax")->Attr("Tuple"))
      ->Call(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleTypeNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TupleTypeDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
