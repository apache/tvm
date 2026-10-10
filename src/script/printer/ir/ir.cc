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
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/op.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/printer.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> IRModuleDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* mod =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IRModuleNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ffi::String module_name = d->GetExtraConfig<ffi::String>("ir.module_name", "Module");
  IdDoc module_id = d->AllocId(module_name);
  ExtraStateScope<ffi::Optional<IdDoc>> module_scope(d, "ir.module_id", module_id);
  ExtraStateScope<ffi::Optional<IRModule>> module(d, "ir.module", ffi::GetRef<IRModule>(mod));
  auto body = d->WithDocScope([&]() {
    if (!mod->attrs->dict.empty()) {
      ffi::Array<ExprDoc> keys;
      ffi::Array<ExprDoc> values;
      std::vector<std::pair<ffi::String, ffi::Any>> attrs(mod->attrs->dict.begin(),
                                                          mod->attrs->dict.end());
      std::sort(attrs.begin(), attrs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      for (const auto& [key, value] : attrs) {
        keys.push_back(LiteralDoc::Str(key, std::nullopt));
        values.push_back(AnyValue(d, value));
      }
      d->Emit(ExprStmtDoc(NamespaceDoc("ir")->Attr("module_attrs")->Call({DictDoc(keys, values)})),
              ffi::GetRef<ffi::ObjectRef>(mod));
    }
    if (!mod->global_infos.empty()) {
      ffi::Array<ExprDoc> keys;
      ffi::Array<ExprDoc> values;
      std::vector<std::pair<ffi::String, ffi::Array<GlobalInfo>>> infos(mod->global_infos.begin(),
                                                                        mod->global_infos.end());
      std::sort(infos.begin(), infos.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      for (const auto& [key, entries] : infos) {
        ffi::Array<ExprDoc> items;
        for (const GlobalInfo& entry : entries) {
          items.push_back(AnyValue(d, entry));
        }
        keys.push_back(LiteralDoc::Str(key, std::nullopt));
        values.push_back(ListDoc(items));
      }
      d->Emit(ExprStmtDoc(
                  NamespaceDoc("ir")->Attr("module_global_infos")->Call({DictDoc(keys, values)})),
              ffi::GetRef<ffi::ObjectRef>(mod));
    }
    for (const auto& [gv, func] : mod->functions) {
      ExtraStateScope<ffi::Optional<ffi::String>> name(d, "ir.function_name", gv->name_hint);
      if (auto value = d->Translate(func)) {
        d->Emit(AssignDoc(IdDoc(gv->name_hint), value.value(), std::nullopt), func);
      }
    }
  });
  d->Emit(ClassDoc(module_id, {NamespaceDoc("ir")->Attr("ir_module")}, ToStmtDocArray(body)),
          ffi::GetRef<ffi::ObjectRef>(mod));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterNamespaceAlias("ir.prefix", "I");
  ffi::reflection::TypeAttrDef<IRModuleNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&IRModuleDocTranslate>());
}

ffi::Optional<ExprDoc> DictAttrsDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* attrs =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DictAttrsNode>(input);
  return AnyValue(d, attrs->dict);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<DictAttrsNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&DictAttrsDocTranslate>());
}

ffi::Optional<ExprDoc> GlobalVarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* var =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const GlobalVarNode>(input);
  return GlobalReference(d, var->name_hint);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<GlobalVarNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&GlobalVarDocTranslate>());
}

ffi::Optional<ExprDoc> OpDocTranslate(DocTranslatorObj*, ffi::AnyView input, const ffi::Object*) {
  const auto* op = ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const OpNode>(input);
  return NamespaceDoc("ir")->Attr("Op")->Attr("get")->Call(
      {LiteralDoc::Str(op->name, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<OpNode>().attr(tvm::script::printer::type_attr::kDocTranslate,
                                              FDocTranslate::FromNative<&OpDocTranslate>());
}

ffi::Optional<ExprDoc> FuncTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(input);
  return TypeValue(d, ffi::GetRef<FuncType>(ty), false);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<FuncTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&FuncTypeDocTranslate>());
}

ffi::Optional<ExprDoc> RangeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object*) {
  const auto* range =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RangeNode>(input);
  ExprDoc min = d->Translate(range->min).value();
  ExprDoc end = d->Translate(range->min + range->extent).value();
  return NamespaceDoc("ir")->Attr("Range")->Call({min, end});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<RangeNode>().attr(tvm::script::printer::type_attr::kDocTranslate,
                                                 FDocTranslate::FromNative<&RangeDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
