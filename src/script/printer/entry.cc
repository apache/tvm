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
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/function.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "doc_printer.h"
#include "doc_translator.h"

namespace tvm {
namespace script {
namespace printer {
namespace {

void CollectNamespaces(ffi::AnyView value, std::unordered_set<std::string>* names,
                       std::unordered_set<const ffi::Object*>* visited) {
  const ffi::Object* object = value.as<ffi::Object>();
  if (!object || !visited->insert(object).second) return;
  if (auto name = value.as<NamespaceDocNode>()) names->insert(name->canonical_name);
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto& item : array.value()) CollectNamespaces(item, names, visited);
  } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    for (const auto& [key, item] : map.value()) {
      CollectNamespaces(key, names, visited);
      CollectNamespaces(item, names, visited);
    }
  } else {
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()), [&](const TVMFFIFieldInfo* field) {
          CollectNamespaces(ffi::reflection::FieldGetter(field)(object), names, visited);
        });
  }
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
    std::unordered_set<const ffi::Object*>* active) {
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
        for (const AccessPath& path : context) {
          if (std::none_of(doc->source_paths.begin(), doc->source_paths.end(),
                           [&](const AccessPath& existing) { return SamePath(existing, path); })) {
            doc->source_paths.push_back(path);
          }
          if (origin->second.as<prim::CastNode>()) {
            if (auto call = value.as<CallDocNode>(); call && call->args.size() == 2) {
              call->args[0]->source_paths.push_back(path->Attr("dtype"));
              call->args[1]->source_paths.push_back(path->Attr("value"));
            }
          }
        }
      }
    }
  }
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto& child : array.value()) {
      RecoverDocPaths(child, context, origins, ir_paths, active);
    }
  } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    for (const auto& entry : map.value()) {
      RecoverDocPaths(entry.second, context, origins, ir_paths, active);
    }
  } else {
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()), [&](const TVMFFIFieldInfo* field) {
          if (ffi::String(field->name) == "source_paths") return;
          // Function parameter order is preserved by both function
          // hooks. Restrict each binder and its annotation to that
          // parameter before recovering shared variable occurrences.
          if (auto function = value.as<FunctionDoc>();
              function && ffi::String(field->name) == "args") {
            auto original = origins.find(object);
            if (original != origins.end() && (original->second.as<tirx::PrimFuncNode>() ||
                                              original->second.as<relax::FunctionNode>())) {
              for (size_t i = 0; i < function.value()->args.size(); ++i) {
                std::vector<AccessPath> parameter_paths;
                for (const auto& path : context) {
                  parameter_paths.push_back(path->Attr("params")->ArrayItem(i));
                }
                RecoverDocPaths(function.value()->args[i], parameter_paths, origins, ir_paths,
                                active);
              }
              return;
            }
          }
          RecoverDocPaths(ffi::reflection::FieldGetter(field)(object), context, origins, ir_paths,
                          active);
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

  ffi::Array<ffi::ObjectRef> AnnotatedObjects() const {
    ffi::Array<ffi::ObjectRef> objects;
    ffi::Array<AccessPath> paths;
    for (const auto& [object, message] : config_->obj_to_annotate) objects.push_back(object);
    for (const auto& [path, message] : config_->path_to_annotate) paths.push_back(path);
    return RequestedObjects(objects, paths);
  }

  ffi::Array<ffi::ObjectRef> UnderlinedObjects() const {
    return RequestedObjects(config_->obj_to_underline, config_->path_to_underline);
  }

  ffi::Array<ffi::ObjectRef> AnnotationAncestors(
      const ffi::Array<ffi::ObjectRef>& annotated_objects) const {
    ffi::Array<ffi::ObjectRef> objects = annotated_objects;
    std::unordered_set<const ffi::Object*> annotated;
    for (const auto& object : annotated_objects) annotated.insert(object.get());
    std::vector<AccessPath> targets;
    for (const auto& [object, path] : occurrences_) {
      if (annotated.count(object.get())) targets.push_back(path);
    }
    // Shared identities can occur in several subtrees. Include each original
    // ancestor conservatively, without handing diagnostic traversal to hooks.
    std::unordered_set<const ffi::Object*> seen = annotated;
    for (const auto& [object, path] : occurrences_) {
      if (seen.count(object.get())) continue;
      for (const auto& target : targets) {
        if (path->IsPrefixOf(target)) {
          seen.insert(object.get());
          objects.push_back(object);
          break;
        }
      }
    }
    return objects;
  }

  ffi::Array<ffi::ObjectRef> RequestedObjects(const ffi::Array<ffi::ObjectRef>& requested_objects,
                                              const ffi::Array<AccessPath>& requested_paths) const {
    ffi::Array<ffi::ObjectRef> objects;
    std::unordered_set<const ffi::Object*> seen;
    auto append = [&](const ffi::ObjectRef& object) {
      if (seen.insert(object.get()).second) objects.push_back(object);
    };
    for (const auto& object : requested_objects) append(object);
    // Hooks only need original identities to avoid hiding diagnostics behind
    // compact syntax. Resolve scalar fields to their deepest object owner here;
    // paths, messages, and final Doc-path correspondence remain entry concerns.
    for (const auto& requested : requested_paths) {
      int deepest = -1;
      for (const auto& [object, path] : occurrences_) {
        if (path->IsPrefixOf(requested)) deepest = std::max(deepest, path->depth);
      }
      for (const auto& [object, path] : occurrences_) {
        if (path->depth == deepest && path->IsPrefixOf(requested)) append(object);
      }
    }
    return objects;
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
      std::vector<std::pair<ffi::Any, ffi::Any>> entries(map->begin(), map->end());
      // Match translation's deterministic module member order.
      if (std::all_of(entries.begin(), entries.end(),
                      [](const auto& item) { return item.first.template as<GlobalVarNode>(); })) {
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
          return a.first.template as<GlobalVarNode>()->name_hint <
                 b.first.template as<GlobalVarNode>()->name_hint;
        });
      }
      for (const auto& [key, child] : entries) Index(child, path->MapItem(key), ancestors);
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
  MapDocPaths paths(obj, config);
  auto extra_config = config->extra_config;
  extra_config.Set("script.binding_names", config->binding_names);
  extra_config.Set("script.show_meta", config->show_meta);
  extra_config.Set("script.verbose_expr", config->verbose_expr);
  extra_config.Set("script.syntax_sugar", config->syntax_sugar);
  extra_config.Set("ir.prefix", config->ir_prefix);
  extra_config.Set("script.show_object_address", config->show_object_address);
  extra_config.Set("script.buffer_dtype", ffi::DLDataTypeToString(config->buffer_dtype));
  auto annotated_objects = paths.AnnotatedObjects();
  extra_config.Set("script.annotated_objects", annotated_objects);
  extra_config.Set("script.annotation_ancestors", paths.AnnotationAncestors(annotated_objects));
  extra_config.Set("script.underlined_objects", paths.UnderlinedObjects());
  auto result = details::TranslateWithOptions(obj, std::move(extra_config));
  auto [underline_paths, annotations] = paths.Map(result.doc, result.origins);
  ffi::Array<ffi::Any> header;
  if (result.future_annotations) {
    header.push_back(ffi::String("from __future__ import annotations\n\n"));
  }
  if (result.imports_metadata) {
    header.push_back(
        ffi::String(result.metadata_loader == "tvm"
                        ? "import tvm\n"
                        : "import tvm as " + std::string(result.metadata_loader) + "\n"));
  }
  if (obj.as<IRModuleNode>() || obj.as<tirx::PrimFuncNode>() || obj.as<relax::FunctionNode>()) {
    std::unordered_set<std::string> namespaces;
    std::unordered_set<const ffi::Object*> visited;
    CollectNamespaces(result.doc, &namespaces, &visited);
    for (const char* canonical : {"ir", "tirx", "s_tir", "relax"}) {
      if (!namespaces.count(canonical)) continue;
      ffi::String name = canonical;
      ffi::String alias = name == "ir"     ? config->ir_prefix
                          : name == "tirx" ? config->GetExtraConfig<ffi::String>("tirx.prefix", "T")
                          : name == "s_tir"
                              ? config->GetExtraConfig<ffi::String>("s_tir.prefix", "Ts")
                              : config->GetExtraConfig<ffi::String>("relax.prefix", "R");
      // Header policy is fixed here; comments remain ordinary CommentDocs.
      ffi::String import =
          "from tvm.script import " + std::string(name) + " as " + std::string(alias);
      if (config->GetExtraConfig<bool>("script.comment_imports", false) &&
          !(config->show_meta && name == "tirx")) {
        header.push_back(CommentDoc(import));
      } else {
        header.push_back(ffi::String(std::string(import) + "\n"));
      }
      if (name == "tirx" && config->GetExtraConfig<bool>("script.comment_imports", false)) {
        header.push_back(CommentDoc("from tvm.tirx.layout import Axis"));
      }
    }
    if (!header.empty()) header.push_back(ffi::String("\n"));
  }
  return details::RenderPythonScript(result.doc, config, header, underline_paths, annotations);
}

TVM_FFI_STATIC_INIT_BLOCK() { ffi::reflection::GlobalDef().def("script.printer.Script", Script); }

}  // namespace printer
}  // namespace script
}  // namespace tvm
