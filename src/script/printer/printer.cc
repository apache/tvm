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
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/printer.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "doc_printer.h"

namespace tvm {
namespace script {
namespace printer {
namespace {

// Visit only the completed Doc graph, never the IR carried by literal values.
void VisitDocs(ffi::AnyView value, const std::function<void(const Doc&)>& visit,
               std::unordered_set<const ffi::Object*>* visited) {
  const ffi::Object* object = value.as<ffi::Object>();
  if (!object || !visited->insert(object).second) return;
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto& item : array.value()) VisitDocs(item, visit, visited);
  } else if (auto doc = value.as<Doc>()) {
    visit(doc.value());
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()), [&](const TVMFFIFieldInfo* field) {
          if (ffi::String(field->name) != "source_paths") {
            VisitDocs(ffi::reflection::FieldGetter(field)(object), visit, visited);
          }
        });
  }
}

ffi::Any ReflectedField(ffi::AnyView value, const char* name) {
  ffi::Any result;
  if (const auto* object = value.as<ffi::Object>()) {
    ffi::reflection::ForEachFieldInfo(TVMFFIGetTypeInfo(object->type_index()),
                                      [&](const TVMFFIFieldInfo* field) {
                                        if (ffi::String(field->name) == name)
                                          result = ffi::reflection::FieldGetter(field)(object);
                                      });
  }
  return result;
}

// Metadata references retain their original values in the ordinary origin map.
// Recognize the existing registry[type_key][index] expression rather than a
// reserved spelling, since the generated identifier may have been renamed.
ffi::Map<ffi::String, ffi::Array<ffi::Any>> CollectMetadata(
    const Doc& root, const ffi::Dict<Doc, ffi::ObjectRef>& origins,
    ffi::Optional<IdDoc>* metadata_id) {
  ffi::Map<ffi::String, ffi::Array<ffi::Any>> metadata;
  std::unordered_set<const ffi::Object*> visited;
  VisitDocs(
      root,
      [&](const Doc& doc) {
        auto reference = doc.as<IndexDoc>();
        auto original = origins.Get(doc);
        if (!reference || !original || reference.value()->indices.size() != 1) return;
        auto entries = reference.value()->value.as<IndexDoc>();
        if (!entries || entries.value()->indices.size() != 1) return;
        auto id = entries.value()->value.as<IdDoc>();
        auto key_doc = entries.value()->indices[0].as<LiteralDoc>();
        auto index_doc = reference.value()->indices[0].as<LiteralDoc>();
        if (!id || origins.count(id.value()) || !key_doc || !index_doc) return;
        auto key = key_doc.value()->value.as<ffi::String>();
        const auto* index = index_doc.value()->value.as<IntImmNode>();
        if (!key || key.value() != original.value()->GetTypeKey() || !index) return;
        auto slot = index->value.as<int64_t>();
        if (!slot.has_value() || slot.value() < 0) return;
        if (metadata_id->has_value() && !metadata_id->value().same_as(id.value())) return;
        *metadata_id = id;
        auto values = metadata.Get(key.value()).value_or(ffi::Array<ffi::Any>{});
        while (values.size() <= static_cast<size_t>(slot.value())) values.push_back(nullptr);
        values.Set(slot.value(), original.value());
        metadata.Set(key.value(), values);
      },
      &visited);
  return metadata;
}

ffi::Array<ffi::String> DisplayAliases(const PrinterConfig& config) {
  ffi::Array<ffi::String> aliases;
  for (const auto& [key, fallback] : GetNamespaceAliases()) {
    aliases.push_back(key == "ir.prefix" ? config->ir_prefix
                                         : config->GetExtraConfig<ffi::String>(key, fallback));
  }
  return aliases;
}

void ApplyDisplayNames(const Doc& root, const PrinterConfig& config,
                       const ffi::Dict<Doc, ffi::ObjectRef>& origins,
                       const ffi::Optional<IdDoc>& metadata_id) {
  auto block = root.as<StmtBlockDoc>();
  if (!block || block.value()->stmts.empty()) return;
  Doc definition = block.value()->stmts.back();
  ffi::Optional<IdDoc> root_id;
  ffi::Optional<ffi::String> name;
  if (!config->binding_names.empty()) name = config->binding_names.back();
  if (auto function = definition.as<FunctionDoc>()) {
    root_id = function.value()->name;
    if (name) {
      // Preserve a public function's declared symbol through common reflected
      // attributes, without dispatching on the function's dialect.
      if (auto original = origins.Get(definition)) {
        auto attrs = ReflectedField(ReflectedField(original.value(), "attrs"), "dict");
        if (auto dict = attrs.as<ffi::Map<ffi::String, ffi::Any>>()) {
          if (auto symbol = dict->Get("global_symbol")) {
            TVM_FFI_CHECK(symbol.value().cast<ffi::String>() == name.value(), TypeError)
                << "printer function global_symbol must match its definition name";
          }
        }
      }
    }
  } else if (auto module = definition.as<ClassDoc>()) {
    root_id = module.value()->name;
    if (!name)
      name = config->GetExtraConfig<ffi::Optional<ffi::String>>("ir.module_name", std::nullopt);
  }
  auto reserved = DisplayAliases(config);
  if (auto module_name =
          config->GetExtraConfig<ffi::Optional<ffi::String>>("ir.module_name", std::nullopt)) {
    reserved.push_back(module_name.value());
  }
  if (root_id && name) reserved.push_back(name.value());
  std::vector<IdDoc> identifiers;
  std::unordered_set<std::string> used(reserved.begin(), reserved.end());
  std::unordered_set<const ffi::Object*> visited;
  VisitDocs(
      root,
      [&](const Doc& doc) {
        if (auto id = doc.as<IdDoc>()) {
          identifiers.push_back(id.value());
          used.insert(id.value()->name);
        }
      },
      &visited);
  for (const ffi::String& spelling : reserved) {
    std::string replacement;
    for (size_t suffix = 1;; ++suffix) {
      replacement = std::string(spelling) + "_" + std::to_string(suffix);
      if (!used.count(replacement)) break;
    }
    bool renamed = false;
    for (const IdDoc& id : identifiers) {
      if (id->name == spelling && !(root_id && root_id.value().same_as(id)) &&
          (origins.count(id) || (metadata_id && metadata_id.value().same_as(id)))) {
        id->name = replacement;
        renamed = true;
      }
    }
    if (renamed) used.insert(replacement);
  }
  if (root_id && name) root_id.value()->name = name.value();
}

bool SamePath(const AccessPath& a, const AccessPath& b) {
  return a->depth == b->depth && a->IsPrefixOf(b);
}

// Join the completed Doc tree to IR occurrences through translator-owned
// origins. Ancestor paths disambiguate a shared object used in separate
// statements; unresolved repeated uses retain every candidate path.
void RecoverDocPaths(
    ffi::AnyView value, const std::vector<AccessPath>& enclosing_ir_paths,
    const std::unordered_map<const ffi::Object*, ffi::ObjectRef>& origins,
    const std::unordered_map<const ffi::Object*, std::vector<AccessPath>>& ir_paths,
    std::unordered_set<const ffi::Object*>* active, bool function_parameter = false) {
  const ffi::Object* object = value.as<ffi::Object>();
  if (!object || !active->insert(object).second) return;
  std::vector<AccessPath> context = enclosing_ir_paths;
  if (auto doc = value.as<DocNode>()) {
    auto origin = origins.find(object);
    if (origin != origins.end()) {
      auto candidates = ir_paths.find(origin->second.get());
      if (candidates != ir_paths.end()) {
        context.clear();
        for (const AccessPath& path : candidates->second) {
          if (enclosing_ir_paths.empty() ||
              std::any_of(enclosing_ir_paths.begin(), enclosing_ir_paths.end(),
                          [&](const AccessPath& parent) { return parent->IsPrefixOf(path); })) {
            context.push_back(path);
          }
        }
        if (context.empty()) context = candidates->second;
        // Parameter wrappers provide occurrence context; the name and type
        // carry their own visible ranges.
        if (!(function_parameter && value.as<AssignDocNode>())) {
          for (const AccessPath& path : context) {
            if (std::none_of(
                    doc->source_paths.begin(), doc->source_paths.end(),
                    [&](const AccessPath& existing) { return SamePath(existing, path); })) {
              doc->source_paths.push_back(path);
            }
          }
        }
      }
    }
  }
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto& child : array.value()) {
      RecoverDocPaths(child, context, origins, ir_paths, active, function_parameter);
    }
  } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    for (const auto& entry : map.value()) {
      RecoverDocPaths(entry.second, context, origins, ir_paths, active);
    }
  } else {
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()), [&](const TVMFFIFieldInfo* field) {
          if (ffi::String(field->name) == "source_paths") return;
          ffi::Any children = ffi::reflection::FieldGetter(field)(object);
          bool parameters = value.as<FunctionDocNode>() && ffi::String(field->name) == "args";
          if (parameters) {
            auto function_origin = origins.find(object);
            auto args = children.as<ffi::Array<AssignDoc>>();
            auto params =
                function_origin == origins.end()
                    ? std::nullopt
                    : ReflectedField(function_origin->second, "params").as<ffi::Array<ffi::Any>>();
            bool matched = args && params && args->size() == params->size();
            for (size_t i = 0; matched && i < args->size(); ++i) {
              auto origin = origins.find((*args)[i].get());
              matched =
                  origin != origins.end() && origin->second.get() == (*params)[i].as<ffi::Object>();
            }
            if (matched) {
              // Formal argument order identifies its occurrence even when the
              // same object is also referenced by another parameter's type.
              for (size_t i = 0; i < args->size(); ++i) {
                std::vector<AccessPath> parameter_paths;
                for (const AccessPath& path : context)
                  parameter_paths.push_back(path->Attr("params")->ArrayItem(i));
                RecoverDocPaths((*args)[i], parameter_paths, origins, ir_paths, active, true);
              }
              return;
            }
          }
          RecoverDocPaths(children, context, origins, ir_paths, active, parameters);
        });
  }
  active->erase(object);
}

class MapDocPaths {
 public:
  MapDocPaths(ffi::AnyView root, const PrinterConfig& config) : config_(config) {
    std::unordered_set<const ffi::Object*> ancestors;
    Index(root, AccessPath::Root(), &ancestors);
  }

  std::pair<ffi::Array<AccessPath>, ffi::Map<AccessPath, ffi::String>> Map(
      const Doc& root, const ffi::Dict<Doc, ffi::ObjectRef>& doc_origins) {
    auto underline_paths = config_->path_to_underline;
    auto annotations = config_->path_to_annotate;
    std::unordered_map<const ffi::Object*, std::vector<AccessPath>> paths_by_object;
    for (const auto& occurrence : occurrences_) {
      paths_by_object[occurrence.first.get()].push_back(occurrence.second);
    }
    std::unordered_map<const ffi::Object*, ffi::ObjectRef> origins;
    for (const auto& [doc, original] : doc_origins) {
      origins.insert_or_assign(doc.get(), original);
    }
    std::unordered_set<const ffi::Object*> active;
    RecoverDocPaths(root, {}, origins, paths_by_object, &active);
    // Include every requested IR occurrence, even when its own Doc is absent.
    // The renderer selects the deepest visible prefix and reports the residual
    // path for both object-based and explicit path requests.
    for (const auto& occurrence : occurrences_) {
      for (const auto& target : config_->obj_to_underline) {
        if (target.same_as(occurrence.first)) underline_paths.push_back(occurrence.second);
      }
      for (const auto& [target, message] : config_->obj_to_annotate) {
        if (target.same_as(occurrence.first)) annotations.Set(occurrence.second, message);
      }
    }
    return {underline_paths, annotations};
  }

 private:
  void Index(ffi::AnyView value, AccessPath path,
             std::unordered_set<const ffi::Object*>* ancestors) {
    const ffi::Object* object = value.as<ffi::Object>();
    if (!object || !ancestors->insert(object).second) return;
    ffi::ObjectRef ref = ffi::GetRef<ffi::ObjectRef>(object);
    occurrences_.emplace_back(ref, path);
    if (auto array = value.as<ffi::Array<ffi::Any>>()) {
      for (size_t i = 0; i < array->size(); ++i) Index((*array)[i], path->ArrayItem(i), ancestors);
    } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      for (const auto& [key, child] : map.value()) Index(child, path->MapItem(key), ancestors);
    } else {
      ffi::reflection::ForEachFieldInfo(TVMFFIGetTypeInfo(object->type_index()),
                                        [&](const TVMFFIFieldInfo* field) {
                                          Index(ffi::reflection::FieldGetter(field)(object),
                                                path->Attr(ffi::String(field->name)), ancestors);
                                        });
    }
    ancestors->erase(object);
  }

  /*! \brief Read-only diagnostic requests for this print. */
  PrinterConfig config_;
  /*! \brief Every original object occurrence and its IR access path. */
  std::vector<std::pair<ffi::ObjectRef, AccessPath>> occurrences_;
};

}  // namespace

ffi::String Script(const ffi::ObjectRef& obj, const PrinterConfig& config) {
  ffi::Dict<Doc, ffi::ObjectRef> origins;
  Doc doc = DocTranslate(obj, &origins, config->extra_config);
  auto block = doc.as<StmtBlockDoc>();
  if (block && !config->verbose_expr && !block.value()->stmts.empty()) {
    if (auto expression = block.value()->stmts.back().as<ExprStmtDoc>()) {
      auto original = origins.Get(expression.value()->expr);
      if (original && original.value().same_as(obj)) {
        // Statement emissions retain their own input and child-expression origins.
        block.value()->stmts = {block.value()->stmts.back()};
      }
    }
  }
  ffi::Optional<IdDoc> metadata_id;
  auto metadata = CollectMetadata(doc, origins, &metadata_id);
  ApplyDisplayNames(doc, config, origins, metadata_id);
  ffi::Array<ffi::Any> header;
  if (!metadata.empty()) {
    if (config->show_meta) {
      auto aliases = DisplayAliases(config);
      std::unordered_set<std::string> used(aliases.begin(), aliases.end());
      std::unordered_set<const ffi::Object*> visited;
      VisitDocs(
          doc,
          [&](const Doc& child) {
            if (auto id = child.as<IdDoc>()) used.insert(id.value()->name);
          },
          &visited);
      std::string loader = "tvm";
      for (size_t suffix = 1; used.count(loader); ++suffix)
        loader = "tvm_" + std::to_string(suffix);
      ffi::String json = ffi::json::Stringify(
          ffi::ToJSONGraph(metadata, ffi::json::Object{{"tvm_version", TVM_VERSION}}), 2);
      block.value()->stmts.insert(block.value()->stmts.begin(),
                                  AssignDoc(metadata_id.value(),
                                            IdDoc(loader)
                                                ->Attr("ir")
                                                ->Attr("load_json")
                                                ->Call({LiteralDoc::Str(json, std::nullopt)}),
                                            std::nullopt));
      header.push_back(
          ffi::String(loader == "tvm" ? "import tvm\n" : "import tvm as " + loader + "\n"));
    } else {
      block.value()->stmts.push_back(
          CommentDoc("Metadata omitted. Use show_meta=True in script() method to show it."));
    }
  }
  bool definition = false;
  bool future_annotations = false;
  std::unordered_set<std::string> namespaces;
  std::unordered_set<const ffi::Object*> visited;
  VisitDocs(
      doc,
      [&](const Doc& child) {
        if (auto name = child.as<NamespaceDoc>()) namespaces.insert(name.value()->canonical_name);
        if (auto function = child.as<FunctionDoc>()) {
          definition = true;
          // Defer annotations uniformly, including forward references and type
          // parameters. Header policy depends on syntax, not dialect semantics.
          future_annotations |=
              std::any_of(function.value()->args.begin(), function.value()->args.end(),
                          [](const AssignDoc& arg) { return arg->annotation.has_value(); }) ||
              function.value()->return_type.has_value() || !function.value()->type_params.empty();
        }
        definition |= child.as<ClassDoc>().has_value();
      },
      &visited);
  if (future_annotations) {
    header.insert(header.begin(), ffi::String("from __future__ import annotations\n\n"));
  }
  if (definition) {
    std::vector<std::pair<ffi::String, ffi::String>> aliases(GetNamespaceAliases().begin(),
                                                             GetNamespaceAliases().end());
    std::sort(aliases.begin(), aliases.end());
    for (const auto& [key, fallback] : aliases) {
      std::string canonical = key;
      canonical.resize(canonical.size() - std::string(".prefix").size());
      if (!namespaces.count(canonical)) continue;
      ffi::String alias = canonical == "ir" ? config->ir_prefix
                                            : config->GetExtraConfig<ffi::String>(key, fallback);
      ffi::String import = "from tvm.script import " + canonical + " as " + std::string(alias);
      if (config->GetExtraConfig<bool>("ir.comment_imports", false) && !config->show_meta) {
        header.push_back(CommentDoc(import));
      } else {
        header.push_back(ffi::String(std::string(import) + "\n"));
      }
    }
    if (!header.empty()) header.push_back(ffi::String("\n"));
  }
  MapDocPaths paths(obj, config);
  auto [underline_paths, annotations] = paths.Map(doc, origins);
  return details::RenderPythonScript(doc, config, header, underline_paths, annotations);
}

TVM_FFI_STATIC_INIT_BLOCK() { ffi::reflection::GlobalDef().def("script.printer.Script", Script); }

}  // namespace printer
}  // namespace script
}  // namespace tvm
