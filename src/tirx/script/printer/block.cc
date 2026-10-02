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
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/exec_scope.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::String ScopeIdApi(tirx::ScopeBinding scope) {
  auto [parent, current] = tirx::ScopeBindingToStringPair(scope);
  if (parent == "kernel" && current == "cluster") return "cluster_id";
  if (parent == "kernel" && current == "cta") return "cta_id";
  if (parent == "cluster" && current == "cta") return "cta_id_in_cluster";
  if (parent == "cluster" && current == "cta_pair") return "cta_id_in_pair";
  if (parent == "cta" && current == "warpgroup") return "warpgroup_id";
  if (parent == "cta" && current == "warp") return "warp_id";
  if (parent == "warpgroup" && current == "warp") return "warp_id_in_wg";
  if (parent == "warp" && current == "thread") return "lane_id";
  if (parent == "cta" && current == "thread") return "thread_id";
  if (parent == "warpgroup" && current == "thread") return "thread_id_in_wg";
  TVM_FFI_THROW(ValueError) << "printer unknown scope-id binding " << parent << "/" << current;
}

ExprDoc ScopeExtents(DocTranslatorObj* d, const ffi::Array<PrimExpr>& values) {
  ffi::Array<ExprDoc> items;
  for (const PrimExpr& value : values) items.push_back(d->Translate(value).value());
  return ListDoc(items);
}

ffi::Optional<ExprDoc> ScopeIdDefStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                  const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::ScopeIdDefStmtNode>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  const tirx::ScopeIdDef& def = stmt->def;
  ffi::Array<ExprDoc> lhs;
  for (const PrimVar& var : def->def_ids) lhs.push_back(VarDoc(d, var));
  ffi::Array<ExprDoc> args;
  if (def->scope != tirx::ScopeBinding::kClusterCtaPair && def->extents.has_value()) {
    args.push_back(ScopeExtents(d, def->extents.value()));
  }
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (def->preferred_extents.has_value()) {
    keys.push_back("preferred");
    values.push_back(ScopeExtents(d, def->preferred_extents.value()));
  }
  if (!def->def_ids.empty()) {
    PrimType dtype = def->def_ids[0].ty();
    for (const PrimVar& var : def->def_ids) {
      TVM_FFI_CHECK(var.ty() == dtype, TypeError) << "printer mixed scope-id dtypes";
    }
    if (dtype != PrimType::Int(32)) {
      keys.push_back("dtype");
      values.push_back(LiteralDoc::Str(ffi::DLDataTypeToString(dtype->dtype), std::nullopt));
    }
  }
  d->Emit(AssignDoc(TupleDoc(lhs),
                    NamespaceDoc("tirx")->Attr(ScopeIdApi(def->scope))->Call(args, keys, values),
                    std::nullopt),
          ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::ScopeIdDefStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ScopeIdDefStmtDocTranslate>());
}

ffi::Optional<ExprDoc> ExecScopeDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* scope =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::ExecScopeNode>(input);
  return NamespaceDoc("tirx")
      ->Attr("ExecScope")
      ->Call({LiteralDoc::Str(scope->name(), std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::ExecScopeNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ExecScopeDocTranslate>());
}

ffi::Optional<ExprDoc> ScopeIdDefDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* def =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::ScopeIdDefNode>(input);
  auto [parent, current] = tirx::ScopeBindingToStringPair(def->scope);
  ExprDoc ids = AnyValue(d, def->def_ids);
  ExprDoc extents = AnyValue(d, def->extents);
  ExprDoc preferred = AnyValue(d, def->preferred_extents);
  d->RecordOrigin(ids, def->def_ids);
  if (def->extents.has_value()) d->RecordOrigin(extents, def->extents.value());
  if (def->preferred_extents.has_value())
    d->RecordOrigin(preferred, def->preferred_extents.value());
  return NamespaceDoc("tirx")
      ->Attr("ScopeIdDef")
      ->Call({ids, extents, LiteralDoc::Str(parent, std::nullopt),
              LiteralDoc::Str(current, std::nullopt), preferred});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::ScopeIdDefNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ScopeIdDefDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
