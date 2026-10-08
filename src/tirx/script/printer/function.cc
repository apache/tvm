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
#include <tvm/ir/function.h>
#include <tvm/ir/global_info.h>
#include <tvm/ir/prim/op.h>
#include <tvm/script/printer/printer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../s_tir/script/printer/utils.h"
#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

void PrintFunction(DocTranslatorObj* d, const tirx::FunctionNode* func, ExprDoc decorator,
                   const ffi::String& dialect_attr) {
  VarScope vars(d);

  ffi::String name = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value_or("main");
  FunctionDoc doc(ffi::UnsafeInit{});
  {
    ffi::Array<AssignDoc> args;
    std::unordered_set<const VarNode*> signature_symbols;
    for (const Var& var : func->params) {
      ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
          var->ty, [&](const Var& symbol) -> ffi::Expected<ffi::WalkResult> {
            signature_symbols.insert(symbol.get());
            return ffi::WalkResult::Skip();
          });
    }
    ffi::Array<IdDoc> param_ids;
    for (const Var& var : func->params) {
      param_ids.push_back(VarDoc(d, var, !signature_symbols.count(var.get())));
    }
    size_t param_index = 0;
    for (const Var& var : func->params) {
      IdDoc lhs = param_ids[param_index++];
      // Reuse the captured scalar in dependent annotations instead of creating
      // a new parameter identity after an earlier annotation has read it.
      bool shared_symbol = signature_symbols.count(var.get());
      ExprDoc annotation =
          shared_symbol ? d->Translate(var).value() : d->Translate(var->ty).value();
      d->RecordOrigin(annotation, shared_symbol ? ffi::ObjectRef(var) : ffi::ObjectRef(var->ty));
      AssignDoc argument(lhs, std::nullopt, annotation);
      d->RecordOrigin(argument, var);
      args.push_back(argument);
    }

    auto signature_candidates = CopyImplicitDefs(d);
    for (const Var& var : func->params) signature_candidates.erase(var);
    ffi::Array<ffi::String> decorator_keys;
    ffi::Array<ExprDoc> decorator_values;
    if (!func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      decorator_keys.push_back("private");
      decorator_values.push_back(LiteralDoc::Boolean(true, std::nullopt));
    }
    if (func->attrs->dict.count(tirx::attr::kPersistentKernel)) {
      decorator_keys.push_back("persistent");
      decorator_values.push_back(LiteralDoc::Boolean(true, std::nullopt));
    }
    if (!decorator_keys.empty()) decorator = decorator->Call({}, decorator_keys, decorator_values);
    ffi::Array<StmtDoc> body;
    if (func->body.has_value()) body = Body(func->body.value(), d);
    std::vector<std::pair<ffi::String, ffi::Any>> attrs;
    for (const auto& [key, value] : func->attrs->dict) {
      if (key != tvm::attr::kGlobalSymbol && (dialect_attr.empty() || key != dialect_attr) &&
          key != tirx::attr::kPersistentKernel)
        attrs.emplace_back(key, value);
    }
    if (!attrs.empty()) {
      std::sort(attrs.begin(), attrs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      ffi::Array<ExprDoc> keys;
      ffi::Array<ExprDoc> values;
      for (const auto& [key, value] : attrs) {
        keys.push_back(LiteralDoc::Str(key, std::nullopt));
        values.push_back(AnyValue(d, value));
      }
      body.insert(
          body.begin(),
          ExprStmtDoc(NamespaceDoc("tirx")->Attr("func_attr")->Call({DictDoc(keys, values)})));
    }
    ffi::Optional<ExprDoc> ret_type = std::nullopt;
    if (!func->ret_type.as<MissingType>().has_value() && !IsVoidType(func->ret_type)) {
      ret_type = d->Translate(func->ret_type).value();
    }
    doc = FunctionDoc(IdDoc(name), args, {decorator}, ret_type, body);
    FinalizeFunctionDefinitions(d, signature_candidates, doc);
  }
  vars.Close();
  d->Emit(doc, ffi::GetRef<ffi::ObjectRef>(func));
}

namespace {

ffi::Optional<ExprDoc> TirxFunctionDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object* destination) {
  const auto* func =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::FunctionNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  if (func->attrs->dict.count(tvm::attr::kSTir)) {
    PrintSTirFunction(d, func);
  } else {
    PrintFunction(d, func, NamespaceDoc("tirx")->Attr("function"), "");
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterNamespaceAlias("tirx.prefix", "T");
  ffi::reflection::TypeAttrDef<tirx::FunctionNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TirxFunctionDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
