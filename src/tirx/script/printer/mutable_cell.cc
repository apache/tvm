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
#include <tvm/ir/op.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/op/mutable_cell.h>
#include <tvm/tirx/type.h>

#include "../../../script/printer/ir/utils.h"
#include "../../../script/printer/utils.h"

namespace tvm::script::printer::details {
namespace {

ffi::Optional<ExprDoc> MutableCellTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                   const ffi::Object*) {
  const auto* type =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::MutableCellTypeNode>(
          input);
  ffi::Array<ExprDoc> args{LiteralDoc::DataType(type->element_type->dtype, std::nullopt)};
  if (type->scope != "local") args.push_back(LiteralDoc::Str(type->scope, std::nullopt));
  return NamespaceDoc("tirx")->Attr("MutableCellType")->Call(args);
}

ffi::Optional<ExprDoc> MutableCellAllocDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                    const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (!destination || !destination->IsInstance<VarNode>() || call->attrs.defined() ||
      !call->ty_args.empty() || call->args.size() != 1)
    return RawCall(d, call);
  auto type = call->ty.as<tirx::MutableCellType>();
  Var var = ffi::GetRef<Var>(static_cast<const VarNode*>(destination));
  if (!type || !ffi::StructuralEqual()(var->ty, call->ty)) return RawCall(d, call);
  ExprDoc initial = d->Translate(call->args[0]).value();
  ExprDoc annotation = d->Translate(type.value()->element_type).value();
  IdDoc lhs = VarDoc(d, var);
  if (type.value()->scope == "local" && type.value()->element_type.IsScalar() &&
      annotation.as<AttrAccessDoc>()) {
    d->Emit(AssignDoc(lhs, initial, annotation), ffi::GetRef<Call>(call));
  } else {
    ffi::Array<ffi::String> keys{"value"};
    ffi::Array<ExprDoc> values{initial};
    if (type.value()->scope != "local") {
      keys.push_back("scope");
      values.push_back(LiteralDoc::Str(type.value()->scope, std::nullopt));
    }
    ExprDoc rhs =
        NamespaceDoc("tirx")
            ->Attr("alloc_cell")
            ->Call({LiteralDoc::DataType(type.value()->element_type->dtype, std::nullopt)}, keys,
                   values);
    d->Emit(AssignDoc(lhs, rhs, std::nullopt), ffi::GetRef<Call>(call));
  }
  return std::nullopt;
}

ffi::Optional<ExprDoc> MutableCellLoadDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                   const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (call->args.size() != 1 || !call->args[0].as<VarNode>() || call->attrs.defined() ||
      !call->ty_args.empty())
    return RawCall(d, call);
  return d->Translate(call->args[0]);
}

ffi::Optional<ExprDoc> MutableCellStoreDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                    const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (destination || call->args.size() != 2 || !call->args[0].as<VarNode>() ||
      call->attrs.defined() || !call->ty_args.empty())
    return RawCall(d, call);
  ExprDoc lhs = d->Translate(call->args[0]).value();
  ExprDoc rhs = d->Translate(call->args[1]).value();
  d->Emit(AssignDoc(lhs, rhs, std::nullopt), ffi::GetRef<Call>(call));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::MutableCellTypeNode>().attr(
      type_attr::kDocTranslate, FDocTranslate::FromNative<&MutableCellTypeDocTranslate>());
  OpDef("tirx.mutable_cell_alloc")
      .set_attr<FDocTranslate>(op_attr::kOpCallDocTranslate,
                               FDocTranslate::FromNative<&MutableCellAllocDocTranslate>());
  OpDef("tirx.mutable_cell_load")
      .set_attr<FDocTranslate>(op_attr::kOpCallDocTranslate,
                               FDocTranslate::FromNative<&MutableCellLoadDocTranslate>());
  OpDef("tirx.mutable_cell_store")
      .set_attr<FDocTranslate>(op_attr::kOpCallDocTranslate,
                               FDocTranslate::FromNative<&MutableCellStoreDocTranslate>());
  RegisterScriptRepr<tirx::MutableCellTypeNode>();
}

}  // namespace
}  // namespace tvm::script::printer::details
