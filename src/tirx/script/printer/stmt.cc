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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

ffi::Array<StmtDoc> Body(const Stmt& stmt, DocTranslatorObj* d) {
  return ToStmtDocArray(d->WithDocScope([&]() { d->Translate(stmt); }));
}

namespace {

ExprDoc TensorOperandDoc(DocTranslatorObj* d, const Expr& value) {
  if (const auto* region = value.as<TensorRegionNode>()) {
    ExprDoc doc = TensorRegionValue(d, region, true);
    d->RecordOrigin(doc, value);
    return doc;
  }
  if (const auto* tuple = value.as<TupleNode>()) {
    if (tuple->fields.empty()) return LiteralDoc::None(std::nullopt);
    ffi::Array<ExprDoc> fields;
    for (const auto& field : tuple->fields) fields.push_back(TensorOperandDoc(d, field));
    return TupleDoc(fields);
  }
  return AnyValue(d, value);
}

ffi::Optional<ExprDoc> TensorCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  const auto op = call->op.as_or_throw<Op>();
  op.Validate(call);
  ffi::Array<ExprDoc> args;
  for (const auto& arg : call->args) args.push_back(TensorOperandDoc(d, arg));
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  ffi::reflection::ForEachFieldInfo(
      TVMFFIGetTypeInfo(call->attrs->type_index()), [&](const TVMFFIFieldInfo* field) {
        ffi::Any value = ffi::reflection::FieldGetter(field)(call->attrs);
        if ((field->flags & kTVMFFIFieldFlagBitMaskHasDefault) &&
            ffi::StructuralEqual()(
                value, ffi::AnyView::CopyFromTVMFFIAny(field->default_value_or_factory)))
          return;
        keys.push_back(ffi::String(field->name));
        values.push_back(AnyValue(d, value));
      });
  static const auto& names = Op::GetAttrMap<TScriptPrinterName>("TScriptPrinterName");
  return NamedCallCallee(names[op])->Call(args, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("script.printer.TensorCallDocTranslate", []() {
    return FDocTranslate::FromNative<&TensorCallDocTranslate>();
  });
}

ffi::Optional<ExprDoc> EvaluateDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ExprDoc value = d->Translate(stmt->value).value();
  if (auto call = stmt->value.as<CallNode>();
      call && !call->op.same_as(tirx::tensor_data_ptr_op())) {
    d->Emit(ExprStmtDoc(value), ffi::GetRef<ffi::ObjectRef>(stmt));
  } else {
    d->Emit(ExprStmtDoc(NamespaceDoc("tirx")->Attr("evaluate")->Call({value})),
            ffi::GetRef<ffi::ObjectRef>(stmt));
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<EvaluateNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&EvaluateDocTranslate>());
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

ffi::Optional<ExprDoc> SeqStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  for (size_t i = 0; i < stmt->seq.size(); ++i) {
    d->Translate(stmt->seq[i]);
    if (i + 1 == stmt->seq.size()) continue;
    const auto* alloc = stmt->seq[i].as<BindNode>();
    const auto* allocation = alloc ? alloc->value.as<CallNode>() : nullptr;
    const auto* store = stmt->seq[i + 1].as<TensorStoreNode>();
    auto docs = d->CurrentScopeDocs();
    if (!allocation || !allocation->op.same_as(tirx::alloc_tensor_op()) || !store ||
        !alloc->var.same_as(store->dest) || docs.empty())
      continue;
    auto scalar = docs.back().as<AssignDoc>();
    if (!IsScalarBuffer(d, alloc->var) || !scalar.has_value() ||
        !std::all_of(store->indices.begin(), store->indices.end(), tvm::prim::IsZero))
      continue;
    bool reads_allocation = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        store->value, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          reads_allocation |= var.same_as(alloc->var);
          return ffi::WalkResult::Advance();
        });
    if (reads_allocation) continue;
    size_t before = docs.size();
    d->Translate(stmt->seq[++i]);
    // Only combine a plain adjacent store. Translation may emit prerequisites.
    if (docs.size() == before + 1) {
      if (auto initialization = docs.back().as<AssignDoc>()) {
        scalar.value()->rhs = initialization.value()->rhs;
        // Preserve both statement origins using ordinary annotation/assignment
        // occurrences while retaining the value's more precise child origin.
        d->RecordOrigin(scalar.value()->annotation.value(), ffi::GetRef<Bind>(alloc));
        d->RecordOrigin(scalar.value(), ffi::GetRef<TensorStore>(store));
        docs.pop_back();
      }
    }
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<SeqStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&SeqStmtDocTranslate>());
}

ffi::Optional<ExprDoc> RegionStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  TVM_FFI_CHECK(stmt->result_vars.empty(), ValueError)
      << "RegionStmt with result_vars has no supported outward-result script syntax";

  // Inputs, attributes, and parameter types are evaluated before the body
  // parameters enter scope. Explicit Var constructors preserve their exact types.
  ExprDoc rhs(ffi::UnsafeInit{});
  if (stmt->op.same_as(tirx::device_entry_op())) {
    rhs = NamespaceDoc("tirx")->Attr("device_entry")->Call({});
  } else if (stmt->op.same_as(tirx::launch_thread_op())) {
    rhs = NamespaceDoc("tirx")
              ->Attr("launch_thread")
              ->Call({LiteralDoc::Str(stmt->args[0].as_or_throw<StringImm>()->value, std::nullopt),
                      d->Translate(stmt->args[1].as_or_throw<PrimExpr>()).value()});
  } else if (stmt->op.same_as(tirx::device_context_op())) {
    rhs = NamespaceDoc("tirx")
              ->Attr("device_context")
              ->Call({d->Translate(stmt->args[0]).value(), d->Translate(stmt->args[1]).value()});
  } else if (stmt->op.same_as(tirx::compute_scope_op())) {
    rhs =
        NamespaceDoc("tirx")
            ->Attr("compute_scope")
            ->Call({LiteralDoc::Str(stmt->args[0].as_or_throw<StringImm>()->value, std::nullopt)});
  } else if (stmt->op.same_as(tirx::parallel_launch_op())) {
    rhs = NamespaceDoc("tirx")->Attr("parallel_launch")->Call({});
  } else {
    ffi::Array<ExprDoc> args;
    for (const Expr& arg : stmt->args) args.push_back(d->Translate(arg).value());
    for (size_t i = 0; i < args.size(); ++i) {
      args.Set(i, MaterializeCallArgument(d, stmt->args[i], args[i]));
    }
    ffi::Array<ffi::String> keys;
    ffi::Array<ExprDoc> values;
    if (!stmt->body_params.empty()) {
      ffi::Array<ExprDoc> params;
      for (const Var& param : stmt->body_params) {
        ExprDoc value = NamespaceDoc("tirx")->Attr("Var")->Call(
            {LiteralDoc::Str(param->name, std::nullopt), TypeValue(d, param->ty)});
        d->RecordOrigin(value, param);
        params.push_back(value);
      }
      keys.push_back("body_params");
      values.push_back(ListDoc(params));
    }
    if (!stmt->attrs->dict.empty()) {
      keys.push_back("attrs");
      values.push_back(AnyValue(d, stmt->attrs));
    }
    rhs = NamespaceDoc("tirx")->Attr("region")->Call(
        {LiteralDoc::Str(stmt->op->name, std::nullopt), ListDoc(args)}, keys, values);
  }

  VarScope vars(d);
  ffi::Optional<ExprDoc> lhs = std::nullopt;
  ffi::Array<ExprDoc> params;
  for (const Var& param : stmt->body_params) params.push_back(VarDoc(d, param));
  if (params.size() == 1) {
    lhs = params[0];
  } else if (!params.empty()) {
    lhs = TupleDoc(params);
  }
  auto body = Body(stmt->body, d);
  vars.Close();
  d->Emit(ScopeDoc(lhs, rhs, body, /*allow_concise_scoping=*/false),
          ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<RegionStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&RegionStmtDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
