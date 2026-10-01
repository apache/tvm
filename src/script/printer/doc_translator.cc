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
#include "doc_translator.h"

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/doc_translator.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dialect_prefix.h"

namespace tvm {
namespace script {
namespace printer {

namespace {

// The only mutable structural state. Identifier allocation lives with the
// remapping environment; no statements, scopes, paths, or config live here.
struct VariableEnvironment {
  ffi::Dict<Var, IdDoc> mapped;
  ffi::Dict<Var, IdDoc> implicit_defs;
  VariableEnvironment()
      : allocated{"range",  "False",   "None",     "True",     "and",    "as",   "assert", "async",
                  "await",  "break",   "class",    "continue", "def",    "del",  "elif",   "else",
                  "except", "finally", "for",      "from",     "global", "if",   "import", "in",
                  "is",     "lambda",  "nonlocal", "not",      "or",     "pass", "raise",  "return",
                  "try",    "while",   "with",     "yield"} {}
  std::unordered_set<std::string> allocated;
};

// Bindings and reservations stay stable within each function naming region.
class TranslationEngine : public DocTranslatorObj {
 public:
  explicit TranslationEngine(ffi::Map<ffi::String, ffi::Any> config)
      : DocTranslatorObj(VTable(), std::move(config)) {}

  void InitializeConfig(ffi::Map<ffi::String, ffi::Any> config) {
    InitializeExtraConfig(std::move(config));
  }

  static IdDoc Allocate(DocTranslatorObj* d, ffi::AnyView value) {
    auto* self = static_cast<TranslationEngine*>(d);
    std::string base = value.cast<ffi::String>();
    if (base.empty()) base = "v";
    for (char& c : base)
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    if (std::isdigit(static_cast<unsigned char>(base[0]))) base.insert(base.begin(), '_');
    std::string name = base;
    auto reserved = [&](const std::string& candidate) {
      return std::any_of(self->var_scopes_.begin(), self->var_scopes_.end(),
                         [&](const auto& frame) { return frame.allocated.count(candidate); });
    };
    for (int i = 1; reserved(name); ++i) name = base + '_' + std::to_string(i);
    self->var_scopes_.back().allocated.insert(name);
    return IdDoc(name);
  }

  static IdDoc GetOrAllocate(DocTranslatorObj* d, ffi::AnyView value, bool explicit_def) {
    auto* self = static_cast<TranslationEngine*>(d);
    Var var = value.cast<Var>();
    for (size_t i = self->var_scopes_.size(); i != 0; --i) {
      auto& frame = self->var_scopes_[i - 1];
      auto found = frame.mapped.find(var);
      if (found == frame.mapped.end()) continue;
      IdDoc id = (*found).second;
      if (explicit_def) {
        frame.implicit_defs.erase(var);
      }
      return id;
    }
    IdDoc id = Allocate(d, var->name);
    auto& frame = self->var_scopes_.back();
    frame.mapped.Set(var, id);
    if (!explicit_def) frame.implicit_defs.Set(var, id);
    return id;
  }
  static ffi::Dict<Var, IdDoc> ImplicitDefinitions(DocTranslatorObj* d) {
    return static_cast<TranslationEngine*>(d)->var_scopes_.back().implicit_defs;
  }
  static void BeginScope(DocTranslatorObj* d) {
    static_cast<TranslationEngine*>(d)->var_scopes_.emplace_back();
  }
  static void EndScope(DocTranslatorObj* d) {
    auto* self = static_cast<TranslationEngine*>(d);
    TVM_FFI_CHECK(self->var_scopes_.size() > 1, ValueError)
        << "printer cannot pop the root variable scope";
    auto& local = self->var_scopes_.back();
    auto& parent = self->var_scopes_[self->var_scopes_.size() - 2];
    // Unbound references remain captures of the enclosing naming region.
    for (const auto& [var, id] : local.implicit_defs) {
      parent.mapped.Set(var, id);
      parent.implicit_defs.Set(var, id);
      parent.allocated.insert(id->name);
    }
    self->var_scopes_.pop_back();
  }

 private:
  template <typename Signature, auto Function>
  static auto Adapter() {
    using View = ffi::reflection::NativeFunctionView<Signature>;
    return reinterpret_cast<typename View::ABIType>(
        ffi::Any(View::template FromNative<Function>()).template cast<void*>());
  }
  static const DocTranslatorVTable* VTable() {
    static const DocTranslatorVTable table{
        Adapter<ffi::Optional<ExprDoc>(DocTranslatorObj*, ffi::AnyView, const ffi::Object*),
                [](DocTranslatorObj* d, ffi::AnyView value, const ffi::Object* destination) {
                  return d->DefaultTranslate(value, destination);
                }>(),
        Adapter<IdDoc(DocTranslatorObj*, ffi::AnyView), &Allocate>(),
        Adapter<IdDoc(DocTranslatorObj*, ffi::AnyView, bool), &GetOrAllocate>(),
        Adapter<ffi::Dict<Var, IdDoc>(DocTranslatorObj*), &ImplicitDefinitions>(),
        Adapter<void(DocTranslatorObj*), &BeginScope>(),
        Adapter<void(DocTranslatorObj*), &EndScope>()};
    return &table;
  }
  /*! \brief Owned variable environments, with a persistent root scope. */
  std::vector<VariableEnvironment> var_scopes_{1};
};

details::TranslationResult TranslateComplete(ffi::AnyView obj, TranslationEngine* engine,
                                             ffi::Map<ffi::String, ffi::Any> extra_config) {
  auto binding_names = engine->GetExtraConfig<ffi::Array<ffi::String>>("script.binding_names", {});
  // Reserve the same registered or explicitly configured aliases used by rendering.
  for (const auto& [key, fallback] : GetDialectPrefixes()) {
    if (!extra_config.count(key)) extra_config.Set(key, fallback);
  }
  for (const auto& [key, value] : extra_config) {
    std::string option = key;
    if (option.size() >= 7 && option.compare(option.size() - 7, 7, ".prefix") == 0)
      engine->AllocId(value.cast<ffi::String>());
  }
  for (const ffi::String& name : binding_names) engine->AllocId(name);
  if (binding_names.empty()) {
    if (auto name = engine->GetExtraConfig<ffi::Optional<ffi::String>>("script.module_name",
                                                                       std::nullopt)) {
      engine->AllocId(name.value());
    }
  }
  // Allocate generated names after all configured names. A dialect or module
  // named tvm/metadata must not shadow the loader or metadata registry.
  IdDoc metadata_id = engine->AllocId("metadata");
  IdDoc metadata_loader = engine->AllocId("tvm");
  extra_config.Set("script.metadata_name", metadata_id->name);
  engine->InitializeConfig(std::move(extra_config));
  auto docs = engine->WithDocScope([&]() {
    ffi::Optional<ExprDoc> expression = engine->Translate(obj);
    if (expression.has_value()) {
      if (!engine->GetExtraConfig<bool>("script.verbose_expr", true))
        engine->CurrentScopeDocs().clear();
      engine->Emit(ExprStmtDoc(expression.value()), obj.as<ffi::ObjectRef>());
    }
    if (!binding_names.empty() && !engine->CurrentScopeDocs().empty()) {
      Doc root = engine->CurrentScopeDocs().back();
      ffi::String name = binding_names.back();
      if (auto function = root.as<FunctionDoc>()) {
        function.value()->name = IdDoc(name);
      } else if (auto module = root.as<ClassDoc>()) {
        module.value()->name = IdDoc(name);
      }
    }
  });
  auto root_declarations = engine->WithDocScope([&]() {
    auto pending = engine->GetImplicitDefs();
    std::vector<std::pair<Var, IdDoc>> remaining(pending.begin(), pending.end());
    std::sort(remaining.begin(), remaining.end(),
              [](const auto& a, const auto& b) { return a.second->name < b.second->name; });
    for (const auto& [var, _] : remaining) {
      if (!pending.count(var)) continue;
      if (auto rhs = engine->Translate(var, var.get())) {
        engine->Emit(AssignDoc(engine->VarGetOrAllocId(var, true), rhs.value(), std::nullopt), var);
      }
    }
  });
  if (!root_declarations.empty()) {
    for (const Doc& doc : docs) root_declarations.push_back(doc);
    docs = root_declarations;
  }
  using MetadataMap = ffi::Dict<ffi::String, ffi::List<ffi::Any>>;
  auto metadata = engine->GetOrCreateExtraState<MetadataMap>("ir.metadata_map");
  ffi::Array<StmtDoc> body;
  if (!metadata.empty() && engine->GetExtraConfig<bool>("script.show_meta", true)) {
    // Serialize the whole registry as one graph so aliases shared across
    // entries survive. Immutable maps/arrays preserve the loader's contract.
    ffi::Map<ffi::String, ffi::Any> serialized;
    std::vector<ffi::String> keys;
    for (const auto& [key, entries] : metadata) keys.push_back(key);
    std::sort(keys.begin(), keys.end());
    for (const auto& key : keys) {
      auto entries = metadata[key];
      serialized.Set(key, ffi::Array<ffi::Any>(entries.begin(), entries.end()));
    }
    ffi::String json = ffi::json::Stringify(
        ffi::ToJSONGraph(serialized, ffi::json::Object{{"tvm_version", TVM_VERSION}}), 2);
    body.push_back(AssignDoc(IdDoc(metadata_id->name),
                             IdDoc(metadata_loader->name)
                                 ->Attr("ir")
                                 ->Attr("load_json")
                                 ->Call({LiteralDoc::Str(json, std::nullopt)}),
                             std::nullopt));
  }
  for (const auto& stmt : ToStmtDocArray(docs)) body.push_back(stmt);
  if (!metadata.empty() && !engine->GetExtraConfig<bool>("script.show_meta", true)) {
    body.push_back(
        CommentDoc("Metadata omitted. Use show_meta=True in script() method to show it."));
  }
  return {StmtBlockDoc(body), engine->GetDocOrigins(),
          engine->GetOrCreateExtraState<bool>("script.future_annotations"), metadata_loader->name,
          !metadata.empty() && engine->GetExtraConfig<bool>("script.show_meta", true)};
}

}  // namespace

namespace details {
TranslationResult TranslateWithOptions(ffi::AnyView ir,
                                       ffi::Map<ffi::String, ffi::Any> extra_config) {
  auto engine = ffi::make_object<TranslationEngine>(extra_config);
  return TranslateComplete(ir, engine.get(), std::move(extra_config));
}
}  // namespace details

Doc DocTranslate(ffi::AnyView ir, ffi::Dict<Doc, ffi::ObjectRef>* doc_origins,
                 ffi::Map<ffi::String, ffi::Any> extra_config) {
  auto result = details::TranslateWithOptions(ir, std::move(extra_config));
  if (doc_origins) *doc_origins = std::move(result.origins);
  return result.doc;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<DocTranslatorObj>();
  refl::EnsureTypeAttrColumn(kDocTranslate);
  refl::EnsureTypeAttrColumn(kTensorLoadDocTranslate);
  refl::GlobalDef()
      .def("script.printer.DocTranslate",
           [](ffi::AnyView ir, ffi::Map<ffi::String, ffi::Any> extra_config) {
             return DocTranslate(ir, nullptr, std::move(extra_config));
           })
      .def_method("script.printer.Translate", &DocTranslatorObj::Translate)
      .def_method("script.printer.Emit", &DocTranslatorObj::Emit)
      .def_method("script.printer.VarGetOrAllocId", &DocTranslatorObj::VarGetOrAllocId)
      .def_method("script.printer.GetImplicitDefs", &DocTranslatorObj::GetImplicitDefs)
      .def_method("script.printer.AllocId", &DocTranslatorObj::AllocId)
      .def_method("script.printer.BeginVarScope", &DocTranslatorObj::BeginVarScope)
      .def_method("script.printer.EndVarScope", &DocTranslatorObj::EndVarScope)
      .def_method("script.printer.BeginDocScope", &DocTranslatorObj::BeginDocScope)
      .def_method("script.printer.EndDocScope", &DocTranslatorObj::EndDocScope)
      .def("script.printer.WithDocScope", [](const DocTranslator& d, const ffi::Function& fn) {
        return d->WithDocScope([&]() { fn(); });
      });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
