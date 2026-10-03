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
#include <tvm/s_tir/stmt.h>

#include "../../../script/printer/utils.h"
#include "../../../tirx/script/printer/utils.h"

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
      TVM_FFI_CHECK(rhs->callee.as_or_throw<AttrAccessDoc>()->name == "Tensor", TypeError)
          << "Ts.sblock_alloc_buffer cannot reconstruct this nonrepresentable TensorType";
      const auto* buffer_type = buffer.var()->ty.as<tirx::TensorTypeNode>();
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
      d->Translate(match);
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

ffi::Optional<ExprDoc> MatchBufferRegionDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                     const ffi::Object* destination) {
  const auto* match = ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<
      const s_tir::MatchBufferRegionNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ExprDoc source = d->Translate(match->source).value();
  CallDoc rhs = d->Translate(match->buffer.var()->ty).value().as_or_throw<CallDoc>();
  TVM_FFI_CHECK(rhs->callee.as_or_throw<AttrAccessDoc>()->name == "Tensor", TypeError)
      << "Ts.match_buffer cannot reconstruct this nonrepresentable TensorType";
  rhs->callee = NamespaceDoc("s_tir")->Attr("match_buffer");
  rhs->args.insert(rhs->args.begin(), source);
  IdDoc lhs = VarDoc(d, match->buffer);
  d->Emit(AssignDoc(lhs, rhs, std::nullopt), ffi::GetRef<ffi::ObjectRef>(match));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<s_tir::MatchBufferRegionNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&MatchBufferRegionDocTranslate>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterScriptRepr<s_tir::MatchBufferRegionNode>();
  RegisterScriptRepr<s_tir::SBlockNode>();
  RegisterScriptRepr<s_tir::SBlockRealizeNode>();
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
