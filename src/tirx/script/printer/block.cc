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
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/exec_scope.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Array<StmtDoc> SBlockBody(DocTranslatorObj* d, const s_tir::SBlockNode* block,
                               const s_tir::SBlockRealizeNode* realize) {
  auto docs = d->WithDocScope([&]() {
    TVM_FFI_CHECK(!realize || realize->iter_values.size() == block->iter_vars.size(), TypeError)
        << "SBlockRealize binding count must match its iter vars";
    ffi::Array<AssignDoc> axes;
    for (size_t i = 0; i < block->iter_vars.size(); ++i) {
      const tirx::IterVar& iter = block->iter_vars[i];
      ffi::String kind;
      switch (iter->iter_type) {
        case tirx::IterVarType::kDataPar:
          kind = "spatial";
          break;
        case tirx::IterVarType::kCommReduce:
          kind = "reduce";
          break;
        case tirx::IterVarType::kOrdered:
          kind = "scan";
          break;
        case tirx::IterVarType::kOpaque:
          kind = "opaque";
          break;
        default:
          TVM_FFI_THROW(TypeError) << "printer unsupported SBlock iter var kind";
      }
      ExprDoc domain =
          prim::is_zero(iter->dom->min) && iter->dom->min.ty() == iter->dom->extent.ty()
              ? d->Translate(iter->dom->extent).value()
              : NamespaceDoc("ir")
                    ->Attr("Range")
                    ->Attr("from_min_extent")
                    ->Call({d->Translate(iter->dom->min).value(),
                            d->Translate(iter->dom->extent).value()});
      ffi::Array<ExprDoc> args = {domain};
      if (realize) args.push_back(d->Translate(realize->iter_values[i]).value());
      IdDoc lhs = VarDoc(d, iter->var);
      ffi::Array<ffi::String> keys;
      ffi::Array<ExprDoc> values;
      if (iter->var.ty() != PrimType::Int(32)) {
        keys.push_back("dtype");
        values.push_back(
            LiteralDoc::Str(ffi::DLDataTypeToString(iter->var.ty()->dtype), std::nullopt));
      }
      axes.push_back(
          AssignDoc(lhs, NamespaceDoc("s_tir")->Attr("axis")->Attr(kind)->Call(args, keys, values),
                    std::nullopt));
    }
    for (size_t i = 0; i < axes.size(); ++i) d->Emit(axes[i], block->iter_vars[i]);
    if (realize && !tvm::prim::is_one(realize->predicate)) {
      d->Emit(ExprStmtDoc(NamespaceDoc("s_tir")->Attr("where")->Call(
                  {d->Translate(realize->predicate).value()})),
              realize->predicate);
    }
    ffi::Array<ExprDoc> reads;
    for (const TensorRegion& region : block->reads) reads.push_back(d->Translate(region).value());
    d->Emit(ExprStmtDoc(NamespaceDoc("s_tir")->Attr("reads")->Call(reads)), block->reads);
    ffi::Array<ExprDoc> writes;
    for (const TensorRegion& region : block->writes) writes.push_back(d->Translate(region).value());
    d->Emit(ExprStmtDoc(NamespaceDoc("s_tir")->Attr("writes")->Call(writes)), block->writes);
    if (!block->annotations.empty()) {
      d->Emit(
          ExprStmtDoc(
              NamespaceDoc("s_tir")->Attr("sblock_attr")->Call({AnyValue(d, block->annotations)})),
          block->annotations);
    }
    for (const tirx::BufferVar& buffer : block->alloc_buffers) {
      CallDoc rhs = d->Translate(buffer.var()->ty).value().as_or_throw<CallDoc>();
      TVM_FFI_CHECK(rhs->callee.as_or_throw<AttrAccessDoc>()->name == "Buffer", TypeError)
          << "Ts.sblock_alloc_buffer cannot reconstruct this nonrepresentable BufferType";
      const auto* buffer_type = buffer.var()->ty.as<tirx::BufferTypeNode>();
      TVM_FFI_CHECK(
          buffer_type->allocated_addr.empty() ||
              (buffer_type->storage_scope != "global" && buffer_type->storage_scope != "shared" &&
               buffer_type->storage_scope != "shared.dyn" && buffer_type->storage_scope != "local"),
          TypeError)
          << "Ts.sblock_alloc_buffer does not accept allocated_addr in "
          << buffer_type->storage_scope;
      rhs->callee = NamespaceDoc("s_tir")->Attr("sblock_alloc_buffer");
      IdDoc lhs = VarDoc(d, buffer);
      d->Emit(AssignDoc(lhs, rhs, std::nullopt), ffi::GetRef<ffi::ObjectRef>(buffer.get()));
    }
    for (const s_tir::MatchBufferRegion& match : block->match_buffers) {
      ExprDoc source = d->Translate(match->source).value();
      CallDoc rhs = d->Translate(match->buffer.var()->ty).value().as_or_throw<CallDoc>();
      TVM_FFI_CHECK(rhs->callee.as_or_throw<AttrAccessDoc>()->name == "Buffer", TypeError)
          << "Ts.match_buffer cannot reconstruct this nonrepresentable BufferType";
      rhs->callee = NamespaceDoc("s_tir")->Attr("match_buffer");
      rhs->args.insert(rhs->args.begin(), source);
      IdDoc lhs = VarDoc(d, match->buffer);
      d->Emit(AssignDoc(lhs, rhs, std::nullopt), ffi::GetRef<ffi::ObjectRef>(match.get()));
    }
    if (block->init.has_value()) {
      d->Emit(ScopeDoc(std::nullopt, NamespaceDoc("s_tir")->Attr("init")->Call({}),
                       Body(block->init.value(), d)),
              block->init.value());
    }
    d->Translate(block->body);
  });
  return ToStmtDocArray(docs);
}

ffi::Optional<ExprDoc> SBlockRealizeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object* destination) {
  const auto* realize =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const s_tir::SBlockRealizeNode>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  const s_tir::SBlockNode* block = realize->block.get();
  d->Emit(ScopeDoc(std::nullopt,
                   NamespaceDoc("s_tir")->Attr("sblock")->Call(
                       {LiteralDoc::Str(block->name_hint, std::nullopt)}),
                   SBlockBody(d, block, realize)),
          ffi::GetRef<ffi::ObjectRef>(realize));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<s_tir::SBlockRealizeNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&SBlockRealizeDocTranslate>());
}

ffi::Optional<ExprDoc> SBlockDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object* destination) {
  const auto* block =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const s_tir::SBlockNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(ScopeDoc(std::nullopt,
                   NamespaceDoc("s_tir")->Attr("sblock")->Call(
                       {LiteralDoc::Str(block->name_hint, std::nullopt)}, {"no_realize"},
                       {LiteralDoc::Boolean(true, std::nullopt)}),
                   SBlockBody(d, block, nullptr)),
          ffi::GetRef<ffi::ObjectRef>(block));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<s_tir::SBlockNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&SBlockDocTranslate>());
}

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
