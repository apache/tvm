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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/function.h>
#include <tvm/script/printer/printer.h>
#include <tvm/tirx/type.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> FunctionDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* func =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::FunctionNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  VarScope vars(d);

  auto binding_name = d->GetOrCreateExtraState<ffi::Optional<ffi::String>>("ir.function_name");
  ExtraStateScope<ffi::Optional<ffi::String>> nested_name(d, "ir.function_name", std::nullopt);
  auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ffi::String name = binding_name.value_or(global_symbol.value_or("main"));
  ffi::Array<AssignDoc> args;
  ffi::Array<IdDoc> param_ids;
  // A prior annotation may depend on a later scalar parameter.
  for (const Var& param : func->params) {
    param_ids.push_back(VarDoc(d, param));
  }
  size_t param_index = 0;
  for (const Var& param : func->params) {
    IdDoc lhs = param_ids[param_index++];
    ffi::Optional<ExprDoc> annotation = std::nullopt;
    if (!param->ty.as<MissingType>().has_value()) annotation = d->Translate(param->ty).value();
    AssignDoc argument(lhs, std::nullopt, annotation);
    d->RecordOrigin(argument, param);
    args.push_back(argument);
  }
  auto signature_candidates = CopyImplicitDefs(d);
  ffi::Optional<ExprDoc> ret_type = std::nullopt;
  if (!func->ret_ty.as<MissingType>().has_value()) {
    ret_type = d->Translate(func->ret_ty).value();
  }
  ffi::Array<ffi::String> decorator_keys;
  ffi::Array<ExprDoc> decorator_values;
  if (!func->is_pure) {
    decorator_keys.push_back("pure");
    decorator_values.push_back(LiteralDoc::Boolean(false, std::nullopt));
  }
  if (!func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
    decorator_keys.push_back("private");
    decorator_values.push_back(LiteralDoc::Boolean(true, std::nullopt));
  }
  ExprDoc decorator = NamespaceDoc("relax")->Attr("function");
  if (!decorator_keys.empty()) decorator = decorator->Call({}, decorator_keys, decorator_values);
  auto docs = d->WithDocScope([&]() {
    std::vector<std::pair<ffi::String, ffi::Any>> attrs;
    for (const auto& [key, value] : func->attrs->dict) {
      if (key != tvm::attr::kGlobalSymbol) attrs.emplace_back(key, value);
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
      d->Emit(ExprStmtDoc(NamespaceDoc("relax")->Attr("func_attr")->Call({DictDoc(keys, values)})),
              ffi::GetRef<ffi::ObjectRef>(func));
    }
    for (const relax::BindingBlock& block : func->body->blocks) d->Translate(block);
    ExprDoc result = d->Translate(func->body->body).value();
    if (auto integer = func->body->body.as<IntImmNode>();
        integer && ffi::StructuralEqual()(integer->ty, func->ret_ty)) {
      // The declared primitive return type reconstructs this literal's dtype.
      result = LiteralDoc::Int(ffi::GetRef<IntImm>(integer), std::nullopt);
    }
    d->Emit(ReturnDoc(result), ffi::GetRef<ffi::ObjectRef>(func));
  });
  auto body = ToStmtDocArray(docs);
  FunctionDoc function(IdDoc(name), args, {decorator}, ret_type, body);
  FinalizeFunctionDefinitions(d, signature_candidates, function);
  if (binding_name && global_symbol && global_symbol.value() != binding_name.value()) {
    function->body.insert(
        function->body.begin(),
        ExprStmtDoc(NamespaceDoc("relax")
                        ->Attr("func_attr")
                        ->Call({DictDoc({LiteralDoc::Str(tvm::attr::kGlobalSymbol, std::nullopt)},
                                        {LiteralDoc::Str(global_symbol.value(), std::nullopt)})})));
  }
  vars.Close();
  d->Emit(function, ffi::GetRef<ffi::ObjectRef>(func));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterNamespaceAlias("relax.prefix", "R");
  ffi::reflection::TypeAttrDef<relax::FunctionNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&FunctionDocTranslate>());
}

ffi::Optional<ExprDoc> ExternFuncDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* func =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::ExternFuncNode>(
          input);
  if (!func->ty.as<relax::FuncTypeNode>()) {
    // The ExternFunc constructor requires a FuncType; retain diagnostic IR in metadata.
    return AddMetadata(d, ffi::GetRef<relax::ExternFunc>(func));
  }
  ffi::Array<ExprDoc> args = {LiteralDoc::Str(func->global_symbol, std::nullopt)};
  if (!ffi::StructuralEqual()(func->ty, relax::ExternFunc(func->global_symbol)->ty)) {
    ExprDoc type = d->Translate(func->ty).value();
    if (auto opaque = func->ty.as<relax::FuncTypeNode>();
        opaque && !opaque->params.has_value() && opaque->ret.as<AnyTypeNode>() && !opaque->purity &&
        !opaque->derive_func.has_value()) {
      type = type->Call({});
    }
    args.push_back(type);
  }
  return NamespaceDoc("relax")->Attr("ExternFunc")->Call(args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::ExternFuncNode>()
      .attr(kDocTranslate, FDocTranslate::FromNative<&ExternFuncDocTranslate>())
      .attr(type_attr::kModuleFunctionOrder, 0);
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
