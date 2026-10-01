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
#include "utils.h"

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/global_info.h>
#include <tvm/ir/op.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace tvm {
namespace script {
namespace printer {
namespace details {

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
  ExprDoc entries_doc = IdDoc(d->GetExtraConfig<ffi::String>(
      "script.metadata_name", "metadata"))[{LiteralDoc::Str(key, std::nullopt)}];
  ExprDoc doc = entries_doc[{LiteralDoc::Int(index, std::nullopt)}];
  if (auto origin = value.as<ffi::ObjectRef>()) d->RecordOrigin(doc, origin.value());
  return doc;
}

IdDoc VarDoc(DocTranslatorObj* d, const Var& var, bool explicit_def) {
  IdDoc doc = IdDoc(d->VarGetOrAllocId(var, explicit_def)->name);
  d->RecordOrigin(doc, var);
  return doc;
}

bool SyntaxSugar(DocTranslatorObj* d) {
  return d->GetExtraConfig<bool>("script.syntax_sugar", true);
}

bool IsAnnotated(DocTranslatorObj* d, const ffi::ObjectRef& object) {
  for (const auto& item :
       d->GetExtraConfig<ffi::Array<ffi::ObjectRef>>("script.annotated_objects", {})) {
    if (item.same_as(object)) return true;
  }
  return false;
}

bool IsUnderlined(DocTranslatorObj* d, const ffi::ObjectRef& object) {
  for (const auto& item :
       d->GetExtraConfig<ffi::Array<ffi::ObjectRef>>("script.underlined_objects", {})) {
    if (item.same_as(object)) return true;
  }
  return false;
}

bool HasAnnotatedDescendant(DocTranslatorObj* d, const ffi::ObjectRef& object) {
  for (const auto& item :
       d->GetExtraConfig<ffi::Array<ffi::ObjectRef>>("script.annotation_ancestors", {})) {
    if (item.same_as(object)) return true;
  }
  return false;
}

ExprDoc GlobalReference(DocTranslatorObj* d, const ffi::String& name) {
  auto names = d->GetExtraConfig<ffi::Array<ffi::String>>("script.binding_names", {});
  return IdDoc(names.empty() ? d->GetExtraConfig<ffi::String>("script.module_name", "Module")
                             : names.back())
      ->Attr(name);
}

namespace {

ExprDoc TypeValueImpl(DocTranslatorObj* d, const Type& type, bool dtype_literal) {
  if (type.as<relax::PackedFuncTypeNode>())
    return NamespaceDoc("relax")->Attr("PackedFunc")->Call({});
  if (type.as<AnyTypeNode>()) return NamespaceDoc("relax")->Attr("Any")->Call({});
  if (auto tuple = type.as<TupleTypeNode>(); tuple && tuple->fields.empty()) {
    return NamespaceDoc("relax")->Attr("Tuple")->Call({});
  }
  if (auto function = type.as<relax::FuncTypeNode>(); function && !function->params.has_value()) {
    ExprDoc doc = d->Translate(type).value();
    return doc.as<AttrAccessDocNode>() ? doc->Call({}) : doc;
  }
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
  if (type.IsMissing()) {
    return NamespaceDoc("ir")->Attr("Type")->Attr("missing")->Call({});
  }
  if (auto pointer = type.as<PointerTypeNode>()) {
    if (auto primitive = pointer->element_type.as<PrimType>();
        primitive.has_value() && primitive.value().IsVoid() && pointer->storage_scope == "global") {
      return NamespaceDoc("tirx")->Attr("handle")->Call({})->Attr("ty");
    }
    return d->Translate(type).value()->Attr("ty");
  }
  ExprDoc doc = d->Translate(type).value();
  if (type.as<relax::TensorTypeNode>() && doc.as<AttrAccessDocNode>()) return doc->Call({});
  return doc;
}

}  // namespace

ExprDoc TypeValue(DocTranslatorObj* d, const Type& type, bool dtype_literal) {
  ExprDoc doc = TypeValueImpl(d, type, dtype_literal);
  d->RecordOrigin(doc, type);
  return doc;
}

// Shared Call takes Expr values rather than a named wrapper's Python sugar.
ExprDoc MaterializeCallArgument(DocTranslatorObj* d, const Expr& arg, ExprDoc doc) {
  if (const auto* tuple = arg.as<TupleNode>()) {
    if (auto fields = doc.as<TupleDocNode>()) {
      ffi::Array<ExprDoc> values;
      for (size_t i = 0; i < tuple->fields.size(); ++i) {
        values.push_back(MaterializeCallArgument(d, tuple->fields[i], fields->elements[i]));
      }
      doc = NamespaceDoc("relax")->Attr("tuple")->Call(values);
    }
  }
  d->RecordOrigin(doc, arg);
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

// The explicit fallback retains every field, including typed attribute objects.
ExprDoc RawCall(DocTranslatorObj* d, const CallNode* call, bool infer_result,
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
  if (!infer_result) {
    keys.push_back("ty");
    values.push_back(TypeValue(d, call->ty));
  }
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
  // A module's existing scoped registry also identifies shared external captures.
  auto module =
      d->GetOrCreateExtraState<ffi::Optional<ffi::Dict<ffi::String, ffi::List<GlobalInfo>>>>(
          "ir.global_info_map");
  bool use_pep695 = !module.has_value() && d->GetExtraConfig<bool>("script.use_pep695", false);
  auto pending = d->GetImplicitDefs();
  std::vector<Var> parameters;
  if (use_pep695) {
    for (const auto& [var, id] : signature) {
      if (pending.count(var) && var->ty.as<PrimTypeNode>()) parameters.push_back(var);
    }
    std::sort(parameters.begin(), parameters.end(),
              [&](const Var& a, const Var& b) { return signature[a]->name < signature[b]->name; });
    if (!parameters.empty()) d->ExchangeExtraState("script.future_annotations", true);
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
