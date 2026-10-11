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
#include <tvm/script/printer/doc_translator.h>
#include <tvm/target/target.h>
#include <tvm/tirx/type.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> PointerTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PointerTypeNode>(input);
  if (auto primitive = ty->element_type.as<PrimType>()) {
    if (primitive.value().IsVoid()) {
      if (ty->storage_scope == "global") return NamespaceDoc("tirx")->Attr("handle");
      return NamespaceDoc("tirx")->Attr("handle")->Call(
          {}, {"storage_scope"}, {LiteralDoc::Str(ty->storage_scope, std::nullopt)});
    }
    ExprDoc element = LiteralDoc::DataType(primitive.value()->dtype, std::nullopt);
    if (ty->storage_scope.empty()) return NamespaceDoc("tirx")->Attr("handle")->Call({element});
    return NamespaceDoc("tirx")->Attr("handle")->Call(
        {element, LiteralDoc::Str(ty->storage_scope, std::nullopt)});
  }
  if (ty->element_type.as<tirx::TensorMapTypeNode>())
    return NamespaceDoc("tirx")->Attr("TensorMap")->Call({});
  return NamespaceDoc("tirx")->Attr("handle")->Call(
      {d->Translate(ty->element_type).value(), LiteralDoc::Str(ty->storage_scope, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<PointerTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&PointerTypeDocTranslate>());
}

ffi::Optional<ExprDoc> TargetDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object*) {
  const auto* target =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TargetNode>(input);
  return NamespaceDoc("tirx")->Attr("target")->Call({AnyValue(d, target->ToConfig())});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TargetNode>().attr(tvm::script::printer::type_attr::kDocTranslate,
                                                  FDocTranslate::FromNative<&TargetDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
