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
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> DataTypeImmDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* imm =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DataTypeImmNode>(input);
  return NamespaceDoc("tirx")->Attr("dtype")->Call(
      {LiteralDoc::DataType(imm->value, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<DataTypeImmNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&DataTypeImmDocTranslate>());
}

ffi::Optional<ExprDoc> GenericConstDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object*) {
  const auto* constant =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const GenericConstNode>(input);
  if (auto dtype = constant->value.as<DLDataType>()) {
    return LiteralDoc::DataType(dtype.value(), std::nullopt);
  }
  static ffi::reflection::TypeAttrColumn column(type_attr::kConstantDocTranslate);
  if (auto hook = column[constant->value.type_index()]; hook != nullptr) {
    return InvokeDocHook(hook, d, input);
  }
  return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<GenericConstNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&GenericConstDocTranslate>());
}

}  // namespace

ExprDoc AnyValue(DocTranslatorObj* d, ffi::AnyView value) {
  if (value.type_index() == ffi::TypeIndex::kTVMFFINone) return LiteralDoc::None(std::nullopt);
  if (value.as<ffi::Module>() || value.as<runtime::Tensor>()) return AddMetadata(d, value);
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    ffi::Array<ExprDoc> elements;
    for (const auto& element : array.value()) elements.push_back(AnyValue(d, element));
    return ListDoc(elements);
  }
  if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    std::vector<std::pair<ffi::Any, ffi::Any>> entries(map->begin(), map->end());
    if (std::all_of(entries.begin(), entries.end(), [](const auto& entry) {
          return entry.first.template as<ffi::String>().has_value();
        })) {
      std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.first.template as_or_throw<ffi::String>() <
               b.first.template as_or_throw<ffi::String>();
      });
    }
    ffi::Array<ExprDoc> keys;
    ffi::Array<ExprDoc> values;
    for (const auto& [key, item] : entries) {
      keys.push_back(AnyValue(d, key));
      values.push_back(AnyValue(d, item));
    }
    return DictDoc(keys, values);
  }
  if (auto string = value.as<ffi::String>()) {
    return LiteralDoc::Str(string.value(), std::nullopt);
  }
  if (auto boolean = value.as<bool>()) {
    return LiteralDoc::Boolean(boolean.value(), std::nullopt);
  }
  if (auto integer = value.as<int64_t>()) {
    return LiteralDoc::Int(integer.value(), std::nullopt);
  }
  if (auto floating = value.as<double>()) {
    return LiteralDoc::Float(floating.value(), std::nullopt);
  }
  if (auto dtype = value.as<DLDataType>()) {
    return LiteralDoc::DataType(dtype.value(), std::nullopt);
  }
  if (const auto* dtype = value.as<DataTypeImmNode>()) {
    return NamespaceDoc("ir")->Attr("dtype")->Call(
        {LiteralDoc::DataType(dtype->value, std::nullopt)});
  }
  // Any-valued fields must retain IR objects; Expr-valued fields convert literals.
  if (const auto* integer = value.as<IntImmNode>()) {
    ExprDoc doc = NamespaceDoc("tirx")->Attr("IntImm")->Call(
        {LiteralDoc::DataType(integer->ty.as_or_throw<PrimType>()->dtype, std::nullopt),
         LiteralDoc::Int(ffi::GetRef<IntImm>(integer), std::nullopt)});
    d->RecordOrigin(doc, ffi::GetRef<IntImm>(integer));
    return doc;
  }
  if (const auto* floating = value.as<FloatImmNode>()) {
    ExprDoc literal = LiteralDoc::Float(floating->value, std::nullopt);
    if (floating->ty.as_or_throw<PrimType>()->dtype == (DLDataType{kDLFloat, 32, 1})) {
      ExprDoc doc = NamespaceDoc("tirx")->Attr("float32")->Call({literal});
      d->RecordOrigin(doc, ffi::GetRef<FloatImm>(floating));
      return doc;
    }
    if (!std::isfinite(floating->value)) {
      // Nonfinite literals render as strings; the scalar helper converts them to a number.
      literal = NamespaceDoc("tirx")->Attr("float64")->Call({literal})->Attr("value");
    }
    ExprDoc doc =
        NamespaceDoc("tirx")
            ->Attr("FloatImm")
            ->Call({LiteralDoc::DataType(floating->ty.as_or_throw<PrimType>()->dtype, std::nullopt),
                    literal});
    d->RecordOrigin(doc, ffi::GetRef<FloatImm>(floating));
    return doc;
  }
  if (const auto* string = value.as<StringImmNode>()) {
    ExprDoc doc =
        NamespaceDoc("ir")->Attr("StringImm")->Call({LiteralDoc::Str(string->value, std::nullopt)});
    d->RecordOrigin(doc, ffi::GetRef<StringImm>(string));
    return doc;
  }
  return d->Translate(value).value();
}

namespace {

ffi::Optional<ExprDoc> ShapeDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                         const ffi::Object*) {
  const auto* shape =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ffi::ShapeObj>(input);
  ffi::Array<ExprDoc> dimensions;
  for (int64_t value : ffi::GetRef<ffi::Shape>(shape)) {
    dimensions.push_back(LiteralDoc::Int(value, std::nullopt));
  }
  return IdDoc("Shape")->Call(dimensions);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<ffi::ShapeObj>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ShapeDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
