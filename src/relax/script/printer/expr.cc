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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "../../../tirx/script/printer/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> TranslateTuple(DocTranslatorObj* d, ffi::AnyView input, const ffi::Object*) {
  const auto* tuple =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleNode>(input);
  if (tuple->fields.empty()) return NamespaceDoc("relax")->Attr("tuple")->Call({});
  ffi::Array<ExprDoc> fields;
  for (const Expr& field : tuple->fields) fields.push_back(d->Translate(field).value());
  return TupleDoc(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleNode>().attr(kDocTranslate,
                                                 FDocTranslate::FromNative<&TranslateTuple>());
}

ffi::Optional<ExprDoc> TranslateTupleGetItem(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* item =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleGetItemNode>(input);
  return d->Translate(item->tuple).value()[{LiteralDoc::Int(item->index, std::nullopt)}];
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleGetItemNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TranslateTupleGetItem>());
}

ffi::Optional<ExprDoc> TranslateRelaxShapeExpr(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* shape =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::ShapeExprNode>(input);
  ffi::Array<ExprDoc> dimensions;
  for (const PrimExpr& dim : shape->values) dimensions.push_back(RelaxShapeDim(d, dim));
  return NamespaceDoc("relax")->Attr("shape")->Call({ListDoc(dimensions)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::ShapeExprNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TranslateRelaxShapeExpr>());
}

ffi::Optional<ExprDoc> TranslateDataflowVar(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  return TranslateVar(d, input, destination);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::DataflowVarNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TranslateDataflowVar>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
