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
#include <tvm/ir/module.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/function.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "../dialect_prefix.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

using GlobalInfoMap = ffi::Dict<ffi::String, ffi::List<GlobalInfo>>;

namespace {

class GlobalInfoScope {
 public:
  GlobalInfoScope(DocTranslatorObj* d, const IRModuleNode* mod) : d_(d) {
    GlobalInfoMap infos;
    for (const auto& [key, entries] : mod->global_infos) {
      infos.Set(key, ffi::List<GlobalInfo>(entries.begin(), entries.end()));
    }
    saved_ = d_->ExchangeExtraState("ir.global_info_map", infos);
  }
  ~GlobalInfoScope() noexcept {
    if (d_) {
      try {
        Close();
      } catch (...) { /* Preserve translation failure. */
      }
    }
  }
  void Close() {
    DocTranslatorObj* d = std::exchange(d_, nullptr);
    d->ExchangeExtraState("ir.global_info_map", std::move(saved_));
  }

 private:
  DocTranslatorObj* d_;
  std::optional<ffi::Any> saved_;
};

ffi::Optional<ExprDoc> IRModuleDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* mod =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IRModuleNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  auto names = d->GetExtraConfig<ffi::Array<ffi::String>>("script.binding_names", {});
  ffi::String module_name =
      names.empty() ? d->GetExtraConfig<ffi::String>("script.module_name", "Module") : names.back();
  d->AllocId(module_name);
  GlobalInfoScope global_infos(d, mod);
  std::vector<std::pair<GlobalVar, BaseFunc>> functions(mod->functions.begin(),
                                                        mod->functions.end());
  auto rank = [](const BaseFunc& func) {
    if (func.as<relax::ExternFuncNode>()) return 0;
    if (func.as<tirx::PrimFuncNode>()) return 1;
    return 2;
  };
  std::sort(functions.begin(), functions.end(), [&](const auto& a, const auto& b) {
    int left = rank(a.second);
    int right = rank(b.second);
    return left == right ? a.first->name_hint < b.first->name_hint : left < right;
  });
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
          ExprDoc item = AnyValue(d, entry);
          if (key == "mesh") {
            if (auto mesh = item.as<CallDoc>(); mesh && mesh.value()->args.size() == 2) {
              if (auto range = mesh.value()->args[1].as<CallDoc>()) {
                range.value()->callee = NamespaceDoc("ir")->Attr("Range");
              }
            }
          }
          items.push_back(item);
        }
        keys.push_back(LiteralDoc::Str(key, std::nullopt));
        values.push_back(ListDoc(items));
      }
      d->Emit(ExprStmtDoc(
                  NamespaceDoc("ir")->Attr("module_global_infos")->Call({DictDoc(keys, values)})),
              ffi::GetRef<ffi::ObjectRef>(mod));
    }
    for (const auto& [gv, func] : functions) {
      if (func.as<relax::ExternFuncNode>()) {
        d->Emit(AssignDoc(IdDoc(gv->name_hint), d->Translate(func).value(), std::nullopt),
                ffi::GetRef<ffi::ObjectRef>(func.get()));
        continue;
      }
      TVM_FFI_CHECK(func.as<tirx::PrimFuncNode>() || func.as<relax::FunctionNode>(), TypeError)
          << "printer IRModule needs a registered TIRx, Relax, or extern function hook";
      d->Translate(func);
      FunctionDoc doc = d->CurrentScopeDocs().back().as_or_throw<FunctionDoc>();
      doc->name = IdDoc(gv->name_hint);
      if (auto symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
          symbol && symbol.value() != gv->name_hint) {
        ExprDoc attr = func.as<tirx::PrimFuncNode>() ? NamespaceDoc("tirx")->Attr("func_attr")
                                                     : NamespaceDoc("relax")->Attr("func_attr");
        doc->body.insert(doc->body.begin(),
                         ExprStmtDoc(attr->Call(
                             {DictDoc({LiteralDoc::Str(tvm::attr::kGlobalSymbol, std::nullopt)},
                                      {LiteralDoc::Str(symbol.value(), std::nullopt)})})));
      }
    }
  });
  global_infos.Close();
  d->Emit(
      ClassDoc(IdDoc(module_name), {NamespaceDoc("ir")->Attr("ir_module")}, ToStmtDocArray(body)),
      ffi::GetRef<ffi::ObjectRef>(mod));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterDialectPrefix("ir.prefix", "I");
  ffi::reflection::TypeAttrDef<IRModuleNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&IRModuleDocTranslate>());
}

ffi::Optional<ExprDoc> DictAttrsDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* attrs =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DictAttrsNode>(input);
  return AnyValue(d, attrs->dict);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<DictAttrsNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&DictAttrsDocTranslate>());
}

ffi::Optional<ExprDoc> GlobalVarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* var =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const GlobalVarNode>(input);
  return GlobalReference(d, var->name_hint);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<GlobalVarNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&GlobalVarDocTranslate>());
}

ffi::Optional<ExprDoc> FuncTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* ty =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(input);
  return TypeValue(d, ffi::GetRef<FuncType>(ty), false);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<FuncTypeNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&FuncTypeDocTranslate>());
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
  ffi::reflection::TypeAttrDef<RangeNode>().attr(kDocTranslate,
                                                 FDocTranslate::FromNative<&RangeDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
