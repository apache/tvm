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
#include <tvm/script/printer/doc_translator.h>

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
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&DataTypeImmDocTranslate>());
}

}  // namespace

namespace {

ffi::Optional<ExprDoc> ArrayDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object*) {
  return d->WithExtraStateScope<bool>(
      "ir.explicit_value_type", true, [&]() -> ffi::Optional<ExprDoc> {
        ffi::Array<ExprDoc> elements;
        for (const auto& element : input.cast<ffi::Array<ffi::Any>>()) {
          elements.push_back(d->Translate(element).value());
        }
        return ListDoc(elements);
      });
}

ffi::Optional<ExprDoc> MapDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  return d->WithExtraStateScope<bool>(
      "ir.explicit_value_type", true, [&]() -> ffi::Optional<ExprDoc> {
        auto map = input.cast<ffi::Map<ffi::Any, ffi::Any>>();
        std::vector<std::pair<ffi::Any, ffi::Any>> entries(map.begin(), map.end());
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
          keys.push_back(d->Translate(key).value());
          values.push_back(d->Translate(item).value());
        }
        return DictDoc(keys, values);
      });
}

ffi::Optional<ExprDoc> MetadataDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object*) {
  return AddMetadata(d, input);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<ffi::ArrayObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&ArrayDocTranslate>());
  ffi::reflection::TypeAttrDef<ffi::MapObj>().attr(tvm::script::printer::type_attr::kDocTranslate,
                                                   FDocTranslate::FromNative<&MapDocTranslate>());
  ffi::reflection::TypeAttrDef<ffi::ModuleObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&MetadataDocTranslate>());
  ffi::reflection::TypeAttrDef<ffi::TensorObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&MetadataDocTranslate>());
}

}  // namespace

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
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&ShapeDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
