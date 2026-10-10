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
#include <tvm/ir/stmt.h>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

ffi::Array<StmtDoc> Body(const Stmt& stmt, DocTranslatorObj* d) {
  return ToStmtDocArray(d->WithDocScope([&]() { d->Translate(stmt); }));
}

namespace {

ffi::Optional<ExprDoc> BindDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->VarGetOrAllocId(stmt->var, false);
  bool existing = !d->GetImplicitDefs().count(stmt->var);
  IdDoc lhs = VarDoc(d, stmt->var);
  auto rhs = d->Translate(stmt->value, stmt->var);
  // None means the child completed emission. A returned expression is still
  // an RHS, so this binding owns the one remaining assignment.
  if (!rhs.has_value()) return std::nullopt;
  ffi::Optional<ExprDoc> annotation = std::nullopt;
  if (!existing) {
    ExprDoc type = d->Translate(stmt->var->ty).value();
    if (auto primitive = stmt->var->ty.as<PrimType>();
        primitive && primitive.value().IsScalableVector()) {
      type = TypeValue(d, stmt->var->ty);
    }
    annotation = NamespaceDoc("tirx")->Attr("let")[{type}];
  }
  d->Emit(AssignDoc(lhs, rhs.value(), annotation), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<BindNode>().attr(kDocTranslate,
                                                FDocTranslate::FromNative<&BindDocTranslate>());
}

ffi::Optional<ExprDoc> ReturnDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(ReturnDoc(d->Translate(stmt->value).value()), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<ReturnNode>().attr(kDocTranslate,
                                                  FDocTranslate::FromNative<&ReturnDocTranslate>());
}

ffi::Optional<ExprDoc> AssertStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ffi::Array<ExprDoc> parts;
  for (const StringImm& part : stmt->message_parts) {
    parts.push_back(LiteralDoc::Str(part->value, std::nullopt));
  }
  d->Emit(
      AssertDoc(d->Translate(stmt->condition).value(),
                TupleDoc({LiteralDoc::Str(stmt->error_kind->value, std::nullopt), ListDoc(parts)})),
      ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<AssertStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&AssertStmtDocTranslate>());
}

ffi::Optional<ExprDoc> WhileDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(WhileDoc(d->Translate(stmt->condition).value(), Body(stmt->body, d)),
          ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<WhileNode>().attr(kDocTranslate,
                                                 FDocTranslate::FromNative<&WhileDocTranslate>());
}

ffi::Optional<ExprDoc> BreakDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BreakNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(BreakDoc(), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<BreakNode>().attr(kDocTranslate,
                                                 FDocTranslate::FromNative<&BreakDocTranslate>());
}

ffi::Optional<ExprDoc> ContinueDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ContinueNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(ContinueDoc(), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<ContinueNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ContinueDocTranslate>());
}

ffi::Optional<ExprDoc> IfDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                      const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ExprDoc condition = d->Translate(stmt->condition).value();
  ffi::Array<StmtDoc> then_body = Body(stmt->then_case, d);
  ffi::Optional<ffi::Array<StmtDoc>> else_body;
  if (stmt->else_case.has_value()) else_body = Body(stmt->else_case.value(), d);
  d->Emit(IfDoc(condition, then_body, else_body), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<IfNode>().attr(kDocTranslate,
                                              FDocTranslate::FromNative<&IfDocTranslate>());
}

}  // namespace
}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
