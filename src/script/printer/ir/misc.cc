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
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/global_info.h>
#include <tvm/runtime/tensor.h>
#include <tvm/target/target.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
  auto tensor = constant->value.as<runtime::Tensor>();
  if (!tensor.has_value()) return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  const runtime::Tensor& data = tensor.value();
  DLDataType dtype = data.DataType();
  auto print_const = [&](ExprDoc value) -> ExprDoc {
    if (constant->ty.as<relax::distributed::DTensorTypeNode>()) {
      return NamespaceDoc("relax")->Attr("dist")->Attr("const")->Call(
          {value, d->Translate(constant->ty).value()});
    }
    return NamespaceDoc("relax")->Attr("const")->Call(
        {value, LiteralDoc::DataType(dtype, std::nullopt)});
  };
  if (data->ndim != 0) return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  if (data->device.device_type != kDLCPU)
    return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  if (dtype.lanes != 1 || dtype.bits == 0 || dtype.bits > 64 || dtype.bits % 8 != 0) {
    return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  }
  alignas(double) uint8_t scalar_bytes[sizeof(double)]{};
  data.CopyToBytes(scalar_bytes, dtype.bits / 8);
  auto read_scalar = [&](auto value) {
    std::memcpy(&value, scalar_bytes, sizeof(value));
    return value;
  };
  ExprDoc scalar(ffi::UnsafeInit{});
  if (dtype == (DLDataType{kDLInt, 8, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int8_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLInt, 16, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int16_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLInt, 32, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int32_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLInt, 64, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int64_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLFloat, 16, 1})) {
    uint16_t bits = read_scalar(uint16_t{});
    uint16_t exponent = (bits >> 10) & 31;
    uint16_t fraction = bits & 1023;
    double value = exponent == 31
                       ? (fraction ? std::numeric_limits<double>::quiet_NaN()
                                   : std::numeric_limits<double>::infinity())
                       : (exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24)
                                        : std::ldexp(static_cast<double>(fraction | 1024),
                                                     static_cast<int>(exponent) - 25));
    if (bits & 0x8000) value = -value;
    scalar = LiteralDoc::Float(value, std::nullopt);
  } else if (dtype == (DLDataType{kDLFloat, 32, 1})) {
    scalar = LiteralDoc::Float(read_scalar(float{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLFloat, 64, 1})) {
    scalar = LiteralDoc::Float(read_scalar(double{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLBool, 8, 1})) {
    scalar = LiteralDoc::Boolean(read_scalar(uint8_t{}), std::nullopt);
  } else {
    return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  }
  return print_const(scalar);
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
  if (auto vdevice = value.as<relax::VDevice>()) {
    return NamespaceDoc("relax")->Attr("vdevice")->Call(
        {NamespaceDoc("tirx")->Attr("target")->Call(
            {AnyValue(d, vdevice.value()->target->ToConfig())})},
        {"vdevice_id", "memory_scope"},
        {LiteralDoc::Int(vdevice.value()->vdevice_id, std::nullopt),
         LiteralDoc::Str(vdevice.value()->memory_scope, std::nullopt)});
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
