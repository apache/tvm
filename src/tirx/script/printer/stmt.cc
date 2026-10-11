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
#include <tvm/script/printer/doc_translator.h>
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

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

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
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&EvaluateDocTranslate>());
}

ffi::Optional<ExprDoc> SeqStmtDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object* destination) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  for (const Stmt& child : stmt->seq) {
    d->Translate(child);
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<SeqStmtNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&SeqStmtDocTranslate>());
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
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : stmt->args) args.push_back(d->Translate(arg).value());
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  bool infer_params = false;
  try {
    auto inferred = GetRegionBodyParams(stmt->op, stmt->args, stmt->attrs);
    infer_params = inferred.size() == stmt->body_params.size();
    for (size_t i = 0; infer_params && i < inferred.size(); ++i) {
      infer_params = ffi::StructuralEqual()(inferred[i]->ty, stmt->body_params[i]->ty);
    }
  } catch (const ffi::Error&) {
    // Explicit parameters preserve regions whose inference is unavailable.
  }
  if (!infer_params) {
    ffi::Array<ExprDoc> params;
    for (const Var& param : stmt->body_params) {
      ExprDoc value = NamespaceDoc("tirx")->Attr("Var")->Call(
          {LiteralDoc::Str("", std::nullopt), TypeValue(d, param->ty)});
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
  ExprDoc rhs(ffi::UnsafeInit{});
  ffi::Optional<ffi::String> name;
  if (Op::HasAttrMap(tvm::script::printer::op_attr::kScriptPrinterName)) {
    auto names =
        Op::GetAttrMap<TScriptPrinterName>(tvm::script::printer::op_attr::kScriptPrinterName);
    if (names.count(stmt->op)) name = names[stmt->op];
  }
  if (name.has_value() && !name.value().empty()) {
    rhs = NamedCallCallee(name.value())->Call(args, keys, values);
  } else {
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
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&RegionStmtDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
