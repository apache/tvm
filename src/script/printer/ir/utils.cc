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
#include "utils.h"

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {
namespace details {

ffi::Optional<ExprDoc> InvokeDocHook(ffi::AnyView hook, DocTranslatorObj* d, ffi::AnyView input,
                                     const ffi::Object* destination) {
  if (hook.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Optional<ExprDoc>>(
               reinterpret_cast<decltype(DocTranslatorVTable::translate)>(hook.cast<void*>())(
                   d, input, destination))
        .value();
  }
  TVM_FFI_CHECK(hook.type_index() == ffi::TypeIndex::kTVMFFIFunction, TypeError)
      << "printer type hook must be a native pointer or ffi.Function";
  ffi::Any binder = destination ? ffi::Any(ffi::GetRef<ffi::ObjectRef>(destination)) : nullptr;
  return hook.cast<ffi::Function>().CallExpected<ffi::Optional<ExprDoc>>(d, input, binder).value();
}

ExprDoc AddMetadata(DocTranslatorObj* d, ffi::Any value) {
  TVM_FFI_CHECK(value != nullptr, TypeError) << "Metadata cannot contain None";
  using MetadataMap = ffi::Dict<ffi::String, ffi::List<ffi::Any>>;
  auto metadata = d->GetOrCreateExtraState<MetadataMap>("ir.metadata_map");
  ffi::String key = value.GetTypeKey();
  auto entries = metadata.Get(key).value_or(ffi::List<ffi::Any>{});
  size_t index = 0;
  for (; index < entries.size(); ++index) {
    if (ffi::AnyEqual()(entries[index], value)) break;
  }
  if (index == entries.size()) {
    entries.push_back(value);
    metadata.Set(key, entries);
  }
  auto metadata_id = d->GetOrCreateExtraState<ffi::Optional<IdDoc>>("ir.metadata_id");
  TVM_FFI_ICHECK(metadata_id.has_value());
  ExprDoc entries_doc = metadata_id.value()[{LiteralDoc::Str(key, std::nullopt)}];
  ExprDoc doc = entries_doc[{LiteralDoc::Int(index, std::nullopt)}];
  if (auto origin = value.as<ffi::ObjectRef>()) d->RecordOrigin(doc, origin.value());
  return doc;
}

IdDoc VarDoc(DocTranslatorObj* d, const Var& var, bool explicit_def) {
  IdDoc doc = IdDoc(d->VarGetOrAllocId(var, explicit_def)->name);
  d->RecordOrigin(doc, var);
  return doc;
}

ExprDoc NamedCallCallee(const ffi::String& canonical_name) {
  std::string name = canonical_name;
  size_t end = name.find('.');
  ExprDoc callee = NamespaceDoc(name.substr(0, end));
  while (end != std::string::npos) {
    size_t start = end + 1;
    end = name.find('.', start);
    callee = callee->Attr(name.substr(start, end - start));
  }
  return callee;
}

ExprDoc GlobalReference(DocTranslatorObj* d, const ffi::String& name) {
  auto module = d->GetOrCreateExtraState<ffi::Optional<IdDoc>>("ir.module_id");
  if (!module.has_value()) {
    module = IdDoc(d->GetExtraConfig<ffi::String>("ir.module_name", "Module"));
    d->SetExtraState("ir.module_id", module);
  }
  return module.value()->Attr(name);
}

namespace {

ExprDoc TypeValueImpl(DocTranslatorObj* d, const Type& type, bool dtype_literal) {
  if (auto primitive = type.as<PrimType>()) {
    ExprDoc dtype = LiteralDoc::DataType(primitive.value()->dtype, std::nullopt);
    return dtype_literal ? dtype : NamespaceDoc("ir")->Attr("PrimType")->Call({dtype});
  }
  if (auto function = type.as<FuncTypeNode>()) {
    ffi::Array<ExprDoc> args;
    for (const Type& arg : function->arg_types) args.push_back(TypeValue(d, arg, false));
    return NamespaceDoc("ir")
        ->Attr("FuncType")
        ->Call({ListDoc(args), TypeValue(d, function->ret_type, false)});
  }
  if (type.as<MissingType>().has_value()) {
    return NamespaceDoc("ir")->Attr("MissingType")->Call({});
  }
  ExtraStateScope<bool> type_value(d, "ir.type_value", true);
  return d->Translate(type).value();
}

}  // namespace

ExprDoc TypeValue(DocTranslatorObj* d, const Type& type, bool dtype_literal) {
  ExprDoc doc = TypeValueImpl(d, type, dtype_literal);
  d->RecordOrigin(doc, type);
  return doc;
}

namespace {

ExprDoc CallAttrsValue(DocTranslatorObj* d, const Attrs& attrs) {
  if (attrs.as<DictAttrsNode>()) return d->Translate(attrs).value();
  std::vector<std::pair<ffi::String, ffi::Any>> fields;
  ffi::reflection::ForEachFieldInfo(
      TVMFFIGetTypeInfo(attrs->type_index()), [&](const TVMFFIFieldInfo* field) {
        fields.emplace_back(ffi::String(field->name), ffi::reflection::FieldGetter(field)(attrs));
      });
  std::sort(fields.begin(), fields.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  for (const auto& [key, value] : fields) {
    if (key == "type_key") return AddMetadata(d, attrs);
    keys.push_back(key);
    values.push_back(AnyValue(d, value));
  }
  ExprDoc inline_doc =
      NamespaceDoc("ir")
          ->Attr("make_node")
          ->Call({LiteralDoc::Str(attrs->GetTypeKey(), std::nullopt)}, keys, values);
  d->RecordOrigin(inline_doc, attrs);
  // These reflected fields have a complete inline representation. Keep them
  // visible in diagnostics and usable when no metadata table is requested.
  return inline_doc;
}

}  // namespace

// Translate operands as IR values when a dialect offers compact expression sugar.
ExprDoc MaterializeCallArgument(DocTranslatorObj* d, const Expr& arg, ExprDoc doc) {
  if (const auto* tuple = arg.as<TupleNode>()) {
    ffi::Array<ExprDoc> fields;
    for (const Expr& field : tuple->fields) {
      fields.push_back(MaterializeCallArgument(d, field, d->Translate(field).value()));
    }
    doc = NamespaceDoc("ir")->Attr("Tuple")->Call({ListDoc(fields)});
  }
  if (const auto* region = arg.as<TensorRegionNode>()) {
    ffi::Array<ExprDoc> ranges;
    for (const Range& range : region->region) {
      ranges.push_back(
          NamespaceDoc("ir")
              ->Attr("Range")
              ->Attr("from_min_extent")
              ->Call({d->Translate(range->min).value(), d->Translate(range->extent).value()}));
    }
    doc = NamespaceDoc("ir")
              ->Attr("TensorRegion")
              ->Call({d->Translate(region->source).value(), ListDoc(ranges)}, {"ty"},
                     {TypeValue(d, region->ty, false)});
  }
  d->RecordOrigin(doc, arg);
  return doc;
}

// The explicit fallback retains every field, including typed attribute objects.
ExprDoc RawCall(DocTranslatorObj* d, const CallNode* call,
                ffi::Optional<ffi::Array<ExprDoc>> translated_args) {
  ffi::Optional<Op> op = call->op.as<Op>();
  ffi::Array<ExprDoc> args;
  if (translated_args.has_value()) {
    args = translated_args.value();
  } else {
    for (const Expr& arg : call->args) args.push_back(d->Translate(arg).value());
  }
  for (size_t i = 0; i < args.size(); ++i) {
    args.Set(i, MaterializeCallArgument(d, call->args[i], args[i]));
  }
  ExprDoc callee = op.has_value()                 ? LiteralDoc::Str(op.value()->name, std::nullopt)
                   : call->op.as<StringImmNode>() ? AnyValue(d, call->op)
                                                  : d->Translate(call->op).value();
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (call->attrs.defined()) {
    keys.push_back("attrs");
    values.push_back(CallAttrsValue(d, call->attrs));
  }
  if (!call->ty_args.empty()) {
    ffi::Array<ExprDoc> types;
    for (const Type& type : call->ty_args) types.push_back(TypeValue(d, type, false));
    keys.push_back("ty_args");
    values.push_back(ListDoc(types));
  }
  keys.push_back("ty");
  values.push_back(TypeValue(d, call->ty));
  return NamespaceDoc("ir")->Attr("Call")->Call({callee, ListDoc(args)}, keys, values);
}

// The query aliases the active frame; candidate classification must own a copy.
ffi::Dict<Var, IdDoc> CopyImplicitDefs(DocTranslatorObj* d) {
  ffi::Dict<Var, IdDoc> candidates;
  for (const auto& [var, id] : d->GetImplicitDefs()) candidates.Set(var, id);
  return candidates;
}

void FinalizeFunctionDefinitions(DocTranslatorObj* d, const ffi::Dict<Var, IdDoc>& signature,
                                 const FunctionDoc& function) {
  auto module = d->GetOrCreateExtraState<ffi::Optional<IRModule>>("ir.module");
  auto pending = d->GetImplicitDefs();
  std::vector<Var> parameters;
  if (!module.has_value()) {
    for (const auto& [var, id] : signature) {
      if (pending.count(var) && var->ty.as<PrimTypeNode>()) parameters.push_back(var);
    }
    std::sort(parameters.begin(), parameters.end(),
              [&](const Var& a, const Var& b) { return signature[a]->name < signature[b]->name; });
    for (const Var& var : parameters) {
      IdDoc id = VarDoc(d, var);
      if (var->ty.as_or_throw<PrimType>()->dtype == (DLDataType{kDLInt, 64, 1})) {
        function->type_params.push_back(id);
      } else {
        function->type_params.push_back(AssignDoc(id, std::nullopt, d->Translate(var->ty).value()));
      }
    }
  }
}

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
