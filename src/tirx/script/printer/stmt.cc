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
#include <tvm/tirx/attrs.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tile_primitive.h>

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

ffi::Array<StmtDoc> Body(const tirx::Stmt& stmt, DocTranslatorObj* d) {
  return ToStmtDocArray(d->WithDocScope([&]() { d->Translate(stmt); }));
}

namespace {

ffi::Optional<ExprDoc> TilePrimitiveCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                     const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::TilePrimitiveCallNode>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  static const OpAttrMap<TScriptPrinterName>& names =
      Op::GetAttrMap<TScriptPrinterName>("TScriptPrinterName");
  TVM_FFI_CHECK(names.count(stmt->op), TypeError)
      << "printer tile primitive has no canonical script name: " << stmt->op->name;
  std::string name = names[stmt->op];
  TVM_FFI_CHECK(name.find("tirx.tile.") == 0, TypeError)
      << "printer tile primitive name must be in tirx.tile namespace: " << name;
  name.erase(0, 10);
  ffi::String scope;
  switch (stmt->scope->kind) {
    case tirx::ScopeKind::kWarp:
      scope = "warp";
      break;
    case tirx::ScopeKind::kWarpgroup:
      scope = "wg";
      break;
    case tirx::ScopeKind::kCta:
      scope = "cta";
      break;
    case tirx::ScopeKind::kCluster:
      scope = "cluster";
      break;
    default:
      scope = "tile";
  }
  ffi::Array<Doc> args;
  size_t n = stmt->args.size();
  if (n == 2 &&
      (stmt->op->name == "tirx.tile.exp2" || stmt->op->name == "tirx.tile.sqrt" ||
       stmt->op->name == "tirx.tile.reciprocal") &&
      [&]() {
        const auto* dst = stmt->args[0].as<TensorRegionNode>();
        const auto* src = stmt->args[1].as<TensorRegionNode>();
        return dst && src && dst->source.same_as(src->source) &&
               ffi::StructuralEqual()(dst->region, src->region);
      }()) {
    n = 1;
  }
  std::vector<size_t> arg_order;
  if (stmt->op->name == "tirx.tile.reduce_negate" && n == 5) {
    // The parser API takes reduce_op before axes; the IR stores it last.
    arg_order = {0, 1, 4, 2, 3};
  } else {
    for (size_t i = 0; i < n; ++i) arg_order.push_back(i);
  }
  for (size_t i : arg_order) {
    if (auto op = stmt->args[i].as<Op>()) {
      const std::string& op_name = op.value()->name;
      if (op_name.find("tirx.tile.") == 0) {
        args.push_back(LiteralDoc::Str(op_name.substr(10), std::nullopt));
        continue;
      }
    }
    if (const auto* region = stmt->args[i].as<TensorRegionNode>()) {
      // Tile APIs require a region even when every extent is one. Point
      // indexing would instead construct a TensorLoad and select builtin APIs.
      ExprDoc value = TensorRegionValue(d, region, true);
      d->RecordOrigin(value, ffi::GetRef<TensorRegion>(region));
      args.push_back(value);
    } else {
      args.push_back(AnyValue(d, stmt->args[i]));
    }
  }
  auto dict = [&](const auto& source, bool config = false) -> ffi::Optional<DictDoc> {
    if (source.empty()) return std::nullopt;
    std::vector<std::pair<ffi::String, ffi::Any>> sorted;
    for (const auto& [key, value] : source) sorted.emplace_back(key, value);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    ffi::Array<ExprDoc> keys;
    ffi::Array<ExprDoc> values;
    for (const auto& [key, value] : sorted) {
      keys.push_back(LiteralDoc::Str(key, std::nullopt));
      if (config) {
        if (const auto* string = value.template as<StringImmNode>()) {
          values.push_back(LiteralDoc::Str(string->value, std::nullopt));
          continue;
        }
        if (const auto* integer = value.template as<IntImmNode>()) {
          PrimType type = integer->ty.template as_or_throw<PrimType>();
          if (type == PrimType::Bool()) {
            values.push_back(LiteralDoc::Boolean(integer->value != 0, std::nullopt));
            continue;
          }
          int bits = integer->value >= INT32_MIN && integer->value <= INT32_MAX ? 32 : 64;
          if (type == PrimType::Int(bits)) {
            values.push_back(LiteralDoc::Int(ffi::GetRef<IntImm>(integer), std::nullopt));
            continue;
          }
        }
        if (const auto* floating = value.template as<FloatImmNode>()) {
          if (floating->ty.template as_or_throw<PrimType>() == PrimType::Float(32) &&
              std::isfinite(floating->value)) {
            values.push_back(LiteralDoc::Float(floating->value, std::nullopt));
            continue;
          }
        }
      }
      values.push_back(AnyValue(d, value));
    }
    return DictDoc(keys, values);
  };
  ffi::Optional<ExprDoc> dispatch = std::nullopt;
  if (stmt->dispatch.has_value()) {
    dispatch = LiteralDoc::Str(stmt->dispatch.value(), std::nullopt);
  }
  auto keywords = dict(stmt->config, true);
  if (name == "sqrt_with_scale_bias" || name == "exp_with_scale_bias" ||
      name == "exp2_with_scale_bias" || name == "log2_with_scale_bias") {
    ffi::Array<ExprDoc> keys{LiteralDoc::Str("scale", std::nullopt),
                             LiteralDoc::Str("bias", std::nullopt)};
    ffi::Array<ExprDoc> values{args[2].as_or_throw<ExprDoc>(), args[3].as_or_throw<ExprDoc>()};
    if (keywords.has_value()) {
      for (const auto& key : keywords.value()->keys) keys.push_back(key);
      for (const auto& value : keywords.value()->values) values.push_back(value);
    }
    keywords = DictDoc(keys, values);
    args = {args[0], args[1]};
  }
  d->Emit(OpCallDoc(NamespaceDoc("tirx")->Attr(scope)->Attr(name), args, dict(stmt->workspace),
                    keywords, dispatch),
          ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::TilePrimitiveCallNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TilePrimitiveCallDocTranslate>());
}

ffi::Optional<ExprDoc> EvaluateDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::EvaluateNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ExprDoc value = d->Translate(stmt->value).value();
  if (auto call = stmt->value.as<CallNode>();
      call && !call->op.same_as(tirx::builtin::buffer_data())) {
    d->Emit(ExprStmtDoc(value), ffi::GetRef<ffi::ObjectRef>(stmt));
  } else {
    d->Emit(ExprStmtDoc(NamespaceDoc("tirx")->Attr("evaluate")->Call({value})),
            ffi::GetRef<ffi::ObjectRef>(stmt));
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::EvaluateNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&EvaluateDocTranslate>());
}

ffi::Optional<ExprDoc> ReturnDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::ReturnNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(ReturnDoc(d->Translate(stmt->value).value()), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::ReturnNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ReturnDocTranslate>());
}

ffi::Optional<ExprDoc> BindDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::BindNode>(input);
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
  ffi::reflection::TypeAttrDef<tirx::BindNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BindDocTranslate>());
}

ffi::Optional<ExprDoc> AssertStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::AssertStmtNode>(input);
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
  ffi::reflection::TypeAttrDef<tirx::AssertStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&AssertStmtDocTranslate>());
}

ffi::Optional<ExprDoc> WhileDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::WhileNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(WhileDoc(d->Translate(stmt->condition).value(), Body(stmt->body, d)),
          ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::WhileNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&WhileDocTranslate>());
}

ffi::Optional<ExprDoc> BreakDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::BreakNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(BreakDoc(), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::BreakNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BreakDocTranslate>());
}

ffi::Optional<ExprDoc> ContinueDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::ContinueNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(ContinueDoc(), ffi::GetRef<ffi::ObjectRef>(stmt));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::ContinueNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ContinueDocTranslate>());
}

ffi::Optional<ExprDoc> IfThenElseDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::IfThenElseNode>(input);
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
  ffi::reflection::TypeAttrDef<tirx::IfThenElseNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&IfThenElseDocTranslate>());
}

ffi::Optional<ExprDoc> SeqStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::SeqStmtNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  for (size_t i = 0; i < stmt->seq.size(); ++i) {
    d->Translate(stmt->seq[i]);
    if (i + 1 == stmt->seq.size()) continue;
    const auto* alloc = stmt->seq[i].as<tirx::BindNode>();
    const auto* allocation = alloc ? alloc->value.as<CallNode>() : nullptr;
    const auto* store = stmt->seq[i + 1].as<tirx::TensorStoreNode>();
    auto docs = d->CurrentScopeDocs();
    if (!allocation || !allocation->op.same_as(tirx::builtin::alloc_tensor()) || !store ||
        !alloc->var.same_as(store->buffer) || docs.empty())
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
        d->RecordOrigin(scalar.value()->annotation.value(), ffi::GetRef<tirx::Bind>(alloc));
        d->RecordOrigin(scalar.value(), ffi::GetRef<tirx::TensorStore>(store));
        docs.pop_back();
      }
    }
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::SeqStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&SeqStmtDocTranslate>());
}

ffi::Optional<ExprDoc> RegionStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::RegionStmtNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  TVM_FFI_CHECK(stmt->result_vars.empty(), ValueError)
      << "RegionStmt with result_vars has no supported outward-result script syntax";

  // Inputs, attributes, and parameter types are evaluated before the body
  // parameters enter scope. Explicit Var constructors preserve their exact types.
  ExprDoc rhs(ffi::UnsafeInit{});
  if (stmt->op.same_as(tirx::builtin::device_entry())) {
    rhs = NamespaceDoc("tirx")->Attr("device_entry")->Call({});
  } else if (stmt->op.same_as(tirx::builtin::launch_thread())) {
    rhs = NamespaceDoc("tirx")
              ->Attr("launch_thread")
              ->Call({LiteralDoc::Str(stmt->args[0].as_or_throw<StringImm>()->value, std::nullopt),
                      d->Translate(stmt->args[1].as_or_throw<PrimExpr>()).value()});
  } else if (stmt->op.same_as(tirx::builtin::device_context())) {
    rhs = NamespaceDoc("tirx")
              ->Attr("device_context")
              ->Call({d->Translate(stmt->args[0]).value(), d->Translate(stmt->args[1]).value()});
  } else if (stmt->op.same_as(tirx::builtin::compute_scope())) {
    rhs =
        NamespaceDoc("tirx")
            ->Attr("compute_scope")
            ->Call({LiteralDoc::Str(stmt->args[0].as_or_throw<StringImm>()->value, std::nullopt)});
  } else if (stmt->op.same_as(tirx::builtin::parallel_launch())) {
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
  ffi::reflection::TypeAttrDef<tirx::RegionStmtNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&RegionStmtDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
