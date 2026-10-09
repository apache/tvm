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
#include <tvm/ir/stmt.h>

#include <algorithm>
#include <vector>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

ffi::Array<StmtDoc> Body(const Stmt& stmt, DocTranslatorObj* d) {
  return ToStmtDocArray(d->WithDocScope([&]() { d->Translate(stmt); }));
}

ExprDoc ForIterator(DocTranslatorObj* d, const ForNode* loop, ffi::Array<ffi::String> keys,
                    ffi::Array<ExprDoc> values, const ffi::Map<ffi::String, ffi::Any>& annotations,
                    bool thread_binding) {
  ffi::String method;
  switch (loop->kind) {
    case ForKind::kDefault:
      method = "serial";
      break;
    case ForKind::kParallel:
      method = thread_binding ? "thread_binding" : "parallel";
      break;
    case ForKind::kVectorized:
      method = "vectorized";
      break;
    case ForKind::kUnrolled:
      method = "unroll";
      break;
    default:
      TVM_FFI_THROW(TypeError) << "printer unknown loop kind";
  }
  ExprDoc min = d->Translate(loop->min).value();
  ExprDoc extent = d->Translate(loop->extent).value();
  ExprDoc end = OperationDoc(OperationDocNode::Kind::kAdd, {min, extent});
  if (prim::IsZero(loop->min)) {
    end = extent;
  } else if (loop->min.as<IntImmNode>() && loop->extent.as<IntImmNode>()) {
    // Python integer addition can widen the endpoint before the builder sees
    // it. Keep the IR operation and its dtype for literal bounds as well.
    end = NamespaceDoc("tirx")->Attr("Add")->Call({min, extent});
  }
  if (!annotations.empty()) {
    std::vector<std::pair<ffi::String, ffi::Any>> sorted;
    for (const auto& [key, value] : annotations) sorted.emplace_back(key, value);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    ffi::Array<ExprDoc> annotation_keys;
    ffi::Array<ExprDoc> annotation_values;
    for (const auto& [key, value] : sorted) {
      annotation_keys.push_back(LiteralDoc::Str(key, std::nullopt));
      annotation_values.push_back(AnyValue(d, value));
    }
    keys.push_back("annotations");
    values.push_back(DictDoc(annotation_keys, annotation_values));
  }
  if (loop->step.has_value()) {
    keys.push_back("step");
    values.push_back(d->Translate(loop->step.value()).value());
  }
  ffi::Array<ExprDoc> bounds = {min, end};
  if (thread_binding && prim::IsZero(loop->min) && loop->min.ty() == loop->extent.ty()) {
    bounds = {end};
  }
  ExprDoc callee = NamespaceDoc("tirx")->Attr(method);
  if (loop->kind == ForKind::kDefault && loop->annotations.empty()) {
    callee = IdDoc("range");
    // range's step is positional. Retain even an explicit unit step because
    // it is part of the source For node, unlike an absent step.
    if (loop->step.has_value()) {
      bounds.push_back(values.back());
      keys.pop_back();
      values.pop_back();
    } else if (prim::IsZero(loop->min) && loop->min.ty() == PrimType::Int(32)) {
      bounds.erase(bounds.begin());
    }
  }
  return callee->Call(bounds, keys, values);
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

ffi::Optional<ExprDoc> ForDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination) {
  const auto* loop =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  static ffi::reflection::TypeAttrColumn column(type_attr::kForIteratorDocTranslate);
  auto hook = column[loop->type_index()];
  ExprDoc iterator = hook != nullptr ? InvokeDocHook(hook, d, input).value()
                                     : ForIterator(d, loop, {}, {}, loop->annotations);
  IdDoc var = VarDoc(d, loop->loop_var);
  d->Emit(ForDoc(var, iterator, Body(loop->body, d)), ffi::GetRef<For>(loop));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<ForNode>().attr(kDocTranslate,
                                               FDocTranslate::FromNative<&ForDocTranslate>());
}

ffi::Optional<ExprDoc> EvaluateDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ExtraStateScope<ffi::Optional<Expr>> context(d, "ir.evaluate_value", stmt->value);
  ExprDoc value = d->Translate(stmt->value).value();
  if (!stmt->value.as<CallNode>()) {
    value = NamespaceDoc("tirx")->Attr("evaluate")->Call({value});
  }
  d->Emit(ExprStmtDoc(value), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

ffi::Optional<ExprDoc> SeqStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ffi::Optional<Stmt> previous;
  for (const Stmt& child : stmt->seq) {
    ExtraStateScope<ffi::Optional<Stmt>> sibling(d, "ir.previous_stmt", previous);
    ExtraStateScope<ffi::Optional<Stmt>> current(d, "ir.current_stmt", child);
    d->Translate(child);
    previous = child;
  }
  return std::nullopt;
}

ffi::Optional<ExprDoc> TensorStoreDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object* destination) {
  const auto* store =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorStoreNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  static ffi::reflection::TypeAttrColumn column(type_attr::kTensorStoreDocTranslate);
  if (auto hook = column[store->dest->ty->type_index()]; hook != nullptr) {
    return InvokeDocHook(hook, d, input);
  }
  ExprDoc tensor = d->Translate(store->dest).value();
  ExprDoc value = d->Translate(store->value).value();
  ExprDoc lhs = IndexDoc(tensor, TensorIndices(d, store->indices, true));
  d->Emit(AssignDoc(lhs, value, std::nullopt), ffi::GetRef<ffi::ObjectRef>(store));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::EnsureTypeAttrColumn(type_attr::kTensorStoreDocTranslate);
  ffi::reflection::EnsureTypeAttrColumn(type_attr::kForIteratorDocTranslate);
  ffi::reflection::TypeAttrDef<EvaluateNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&EvaluateDocTranslate>());
  ffi::reflection::TypeAttrDef<SeqStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&SeqStmtDocTranslate>());
  ffi::reflection::TypeAttrDef<TensorStoreNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TensorStoreDocTranslate>());
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
  if (Op::HasAttrMap(op_attr::kRegionDocTranslate) &&
      Op::GetAttrMap<ffi::Any>(op_attr::kRegionDocTranslate).count(stmt->op)) {
    rhs = InvokeDocHook(Op::GetAttrMap<ffi::Any>(op_attr::kRegionDocTranslate)[stmt->op], d, input)
              .value();
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
