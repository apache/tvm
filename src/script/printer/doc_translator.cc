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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/base.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tile_primitive.h>
#include <tvm/tirx/var.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/utils.h"

namespace tvm {
namespace script {
namespace printer {

namespace {

// The only mutable structural state. Identifier allocation lives with the
// remapping environment; no statements, scopes, paths, or config live here.
struct VariableEnvironment {
  ffi::Dict<Var, IdDoc> mapped;
  ffi::Dict<Var, IdDoc> implicit_defs;
  std::unordered_set<std::string> allocated{
      "range", "False", "None",     "True",  "and",    "as",   "assert", "async",  "await",
      "break", "class", "continue", "def",   "del",    "elif", "else",   "except", "finally",
      "for",   "from",  "global",   "if",    "import", "in",   "is",     "lambda", "nonlocal",
      "not",   "or",    "pass",     "raise", "return", "try",  "while",  "with",   "yield"};
};

// The engine owns bindings and their lexical undo records; hooks only allocate
// or promote variables. Reservations survive block exits until function exit.
class TranslationEngine : public DocTranslatorObj {
 public:
  TranslationEngine() : DocTranslatorObj(VTable()) {}

  static ffi::Optional<ExprDoc> TranslateValue(DocTranslatorObj* d, ffi::AnyView value,
                                               const ffi::Object* destination) {
    auto* self = static_cast<TranslationEngine*>(d);
    const ffi::Object* node = value.as<ffi::Object>();
    bool scoped =
        node &&
        (node->IsInstance<tirx::ForNode>() || node->IsInstance<tirx::WhileNode>() ||
         node->IsInstance<tirx::IfThenElseNode>() || node->IsInstance<tirx::AttrStmtNode>() ||
         node->IsInstance<prim::LetNode>() || node->IsInstance<tirx::LambdaExprNode>() ||
         node->IsInstance<tirx::IndexMapNode>() || node->IsInstance<te::CommReducerNode>() ||
         node->IsInstance<te::ReduceNode>() || node->IsInstance<s_tir::SBlockNode>() ||
         node->IsInstance<s_tir::SBlockRealizeNode>());
    // Each branch gets its own lifetime; SeqStmt itself stays transparent.
    if (!self->active_nodes_.empty()) {
      if (auto branch = self->active_nodes_.back().as<tirx::IfThenElseNode>())
        scoped = scoped || node == branch->then_case.get() ||
                 (branch->else_case.has_value() && node == branch->else_case.value().get());
      if (auto branch = self->active_nodes_.back().as<relax::IfNode>())
        scoped = scoped || node == branch->true_branch.get() || node == branch->false_branch.get();
    }
    size_t scope = self->var_scopes_.size() - 1;
    if (scoped) self->binding_undos_.push_back({scope, {}, {}});
    self->active_nodes_.push_back(value);
    ffi::Optional<ExprDoc> result;
    try {
      result = d->DefaultTranslate(value, destination);
    } catch (...) {
      self->active_nodes_.pop_back();
      if (scoped) {
        try {
          self->RestoreBindings();
        } catch (...) { /* Preserve the original error. */
        }
      }
      throw;
    }
    self->active_nodes_.pop_back();
    if (scoped) self->RestoreBindings();
    return result;
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
        self->RememberBinding(i - 1, var);
        frame.implicit_defs.erase(var);
      }
      return id;
    }
    size_t current = self->var_scopes_.size() - 1;
    if (explicit_def)
      self->RememberBinding(current, var);
    else {
      for (auto undo = self->binding_undos_.rbegin(); undo != self->binding_undos_.rend(); ++undo) {
        if (undo->scope == current) {
          undo->created_implicit.insert(var.get());
        }
      }
    }
    ffi::String hint = var->name;
    if (var->ty.as<tirx::BufferTypeNode>() && (hint.empty() || hint == "v")) hint = "buffer";
    if (self->GetExtraConfig<bool>("script.show_object_address", false)) {
      std::ostringstream stream;
      stream << hint << "_" << var.get();
      hint = stream.str();
    }
    IdDoc id = Allocate(d, hint);
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
    // Scalar state is keyed by stable IdDoc. Remove only bindings owned here;
    // captured outer bindings and whole-output accumulators remain untouched.
    auto scalars =
        self->GetOrCreateExtraState<ffi::Dict<IdDoc, bool>>("tirx.buffer_as_mutable_var");
    for (const auto& [var, id] : self->var_scopes_.back().mapped) scalars.erase(id);
    self->var_scopes_.pop_back();
  }

 private:
  struct SavedBinding {
    Var var;
    ffi::Optional<IdDoc> id;
    bool implicit;
    ffi::Optional<bool> scalar_spelling;
  };
  struct BindingUndo {
    size_t scope;
    std::vector<SavedBinding> saved;
    std::unordered_set<const ffi::Object*> created_implicit;
  };
  void RememberBinding(size_t scope, const Var& var) {
    for (auto undo = binding_undos_.rbegin(); undo != binding_undos_.rend(); ++undo) {
      if (undo->scope != scope) continue;
      for (const auto& saved : undo->saved)
        if (saved.var.same_as(var)) return;
      auto& frame = var_scopes_[scope];
      auto previous = frame.mapped.find(var);
      bool introduced_here = undo->created_implicit.count(var.get());
      ffi::Optional<IdDoc> old_id = previous == frame.mapped.end() || introduced_here
                                        ? ffi::Optional<IdDoc>(std::nullopt)
                                        : (*previous).second;
      ffi::Optional<bool> scalar_spelling = std::nullopt;
      auto scalars = GetOrCreateExtraState<ffi::Dict<IdDoc, bool>>("tirx.buffer_as_mutable_var");
      if (old_id.has_value() && scalars.count(old_id.value()))
        scalar_spelling = scalars[old_id.value()];
      undo->saved.push_back(
          {var, old_id, !introduced_here && frame.implicit_defs.count(var) != 0, scalar_spelling});
      return;
    }
  }
  void RestoreBindings() {
    BindingUndo undo = std::move(binding_undos_.back());
    binding_undos_.pop_back();
    auto& frame = var_scopes_[undo.scope];
    auto scalars = GetOrCreateExtraState<ffi::Dict<IdDoc, bool>>("tirx.buffer_as_mutable_var");
    for (auto saved = undo.saved.rbegin(); saved != undo.saved.rend(); ++saved) {
      if (saved->id.has_value()) {
        frame.mapped.Set(saved->var, saved->id.value());
        if (saved->scalar_spelling.has_value())
          scalars.Set(saved->id.value(), saved->scalar_spelling.value());
        else
          scalars.erase(saved->id.value());
        if (saved->implicit)
          frame.implicit_defs.Set(saved->var, saved->id.value());
        else
          frame.implicit_defs.erase(saved->var);
      } else {
        auto current = frame.mapped.find(saved->var);
        if (current != frame.mapped.end()) scalars.erase((*current).second);
        frame.mapped.erase(saved->var);
        frame.implicit_defs.erase(saved->var);
      }
    }
  }
  template <typename Signature, auto Function>
  static auto Adapter() {
    using View = ffi::reflection::NativeFunctionView<Signature>;
    return reinterpret_cast<typename View::ABIType>(
        ffi::Any(View::template FromNative<Function>()).template cast<void*>());
  }
  static const DocTranslatorVTable* VTable() {
    static const DocTranslatorVTable table{
        Adapter<ffi::Optional<ExprDoc>(DocTranslatorObj*, ffi::AnyView, const ffi::Object*),
                &TranslateValue>(),
        Adapter<IdDoc(DocTranslatorObj*, ffi::AnyView), &Allocate>(),
        Adapter<IdDoc(DocTranslatorObj*, ffi::AnyView, bool), &GetOrAllocate>(),
        Adapter<ffi::Dict<Var, IdDoc>(DocTranslatorObj*), &ImplicitDefinitions>(),
        Adapter<void(DocTranslatorObj*), &BeginScope>(),
        Adapter<void(DocTranslatorObj*), &EndScope>()};
    return &table;
  }
  /*! \brief Owned variable environments, with a persistent root scope. */
  std::vector<VariableEnvironment> var_scopes_{1};
  /*! \brief Binding changes restored when each lexical region ends. */
  std::vector<BindingUndo> binding_undos_;
  /*! \brief Owning references to inputs on the active translation stack. */
  std::vector<ffi::Any> active_nodes_;
};

class DocTranslationEngine : public TranslationEngine {
 public:
  DocTranslationEngine(ffi::AnyView root, ffi::Map<ffi::String, ffi::Any> config)
      : TranslationEngine(), root_(root) {
    InitializeConfig(std::move(config));
    use_pep695_ = (root.as<relax::FunctionNode>() || root.as<tirx::PrimFuncNode>()) &&
                  GetExtraConfig<bool>("script.use_pep695", false);
    if (auto branch = root.as<relax::IfNode>()) {
      for (const relax::SeqExpr& arm : {branch->true_branch, branch->false_branch}) {
        if (auto var = arm->body.as<VarNode>(); var && !arm->body.same_as(branch->cond)) {
          branch_local_.insert(var);
        }
      }
    }
    std::unordered_set<const ffi::Object*> ancestors;
    CollectObjects(root, &ancestors);
  }

  bool UsePEP695() const { return use_pep695_; }

  void InitializeConfig(ffi::Map<ffi::String, ffi::Any> config) {
    InitializeExtraConfig(std::move(config));
  }

  void DeclareFreeVariables() {
    std::unordered_set<const ffi::Object*> defined = branch_local_;
    for (const auto& object : objects_) {
      if (auto bind = object.as<tirx::BindNode>()) defined.insert(bind->var.get());
      if (auto scope = object.as<tirx::ScopeIdDefStmtNode>()) {
        for (const PrimVar& var : scope->def->def_ids) defined.insert(var.get());
      }
      if (auto loop = object.as<tirx::ForNode>()) defined.insert(loop->loop_var.get());
      if (auto attr = object.as<tirx::AttrStmtNode>()) {
        if (attr->attr_key == "thread_extent" || attr->attr_key == tirx::attr::virtual_thread) {
          if (auto iter = attr->node.as<tirx::IterVar>()) defined.insert(iter.value()->var.get());
        }
      }
      if (auto func = object.as<tirx::PrimFuncNode>()) {
        for (const auto& var : func->params) defined.insert(var.get());
      }
      if (auto lambda = object.as<tirx::LambdaExprNode>()) {
        for (const auto& var : lambda->vars) defined.insert(var.get());
      }
      if (auto let = object.as<prim::LetNode>()) defined.insert(let->var.get());
      if (auto reducer = object.as<te::CommReducerNode>()) {
        for (const auto& var : reducer->lhs) defined.insert(var.get());
        for (const auto& var : reducer->rhs) defined.insert(var.get());
      }
      if (auto reduce = object.as<te::ReduceNode>()) {
        for (const auto& axis : reduce->axis) defined.insert(axis->var.get());
      }
      if (auto func = object.as<relax::FunctionNode>()) {
        for (const auto& var : func->params) defined.insert(var.get());
      }
      if (auto binding = object.as<relax::BindingNode>()) {
        defined.insert(binding->var.get());
      }
      if (auto block = object.as<s_tir::SBlockNode>()) {
        for (const auto& iter : block->iter_vars) defined.insert(iter->var.get());
        for (const auto& buffer : block->alloc_buffers) defined.insert(buffer.get());
        for (const auto& match : block->match_buffers) defined.insert(match->buffer.get());
      }
    }
    // Classification remains translation-owned for genuinely external captures. A
    // parameter type is a pattern region: its unresolved variables belong to
    // the function's live implicit collection, not this external prelude.
    std::unordered_set<const ffi::Object*> signature_vars;
    std::unordered_map<const ffi::Object*, size_t> signature_owners;
    auto collect_signature = [&](const ffi::Array<Var>& params) {
      std::unordered_set<const ffi::Object*> in_signature;
      for (const Var& param : params) {
        ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
            param->ty, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
              in_signature.insert(var.get());
              return ffi::WalkResult::Advance();
            });
      }
      for (const auto* var : in_signature) ++signature_owners[var];
    };
    for (const auto& object : objects_) {
      if (auto func = object.as<tirx::PrimFuncNode>()) collect_signature(func->params);
      if (auto func = object.as<relax::FunctionNode>()) collect_signature(func->params);
    }
    // Module-level symbols can be shared across function signatures. Keep
    // their identity in the root frame; only one-owner function patterns are
    // left to function-local inference and eligible PEP headers.
    if (!root_.as<IRModuleNode>()) {
      for (const auto& [var, owners] : signature_owners)
        if (owners == 1) signature_vars.insert(var);
    }
    std::vector<Var> free_vars;
    std::unordered_set<const ffi::Object*> allocated;
    for (const auto& object : objects_) {
      if (auto var = object.as<VarNode>(); var && !defined.count(var)) {
        if (signature_vars.count(var) || !allocated.insert(var).second) continue;
        if (var->ty.as<tirx::BufferTypeNode>()) continue;
        Var binding = ffi::GetRef<Var>(var);
        VarGetOrAllocId(binding, true);
        free_vars.push_back(binding);
      }
    }
    // Preserve first-use order while declaring variables referenced by a
    // free variable's type before that variable.  Grouping all scalar shape
    // variables at the top changes the visible order of independent Relax
    // variables and loses the parser's expected declaration grouping.
    std::unordered_map<const ffi::Object*, Var> free_by_object;
    for (const Var& var : free_vars) free_by_object.emplace(var.get(), var);
    std::unordered_set<const ffi::Object*> ordered_objects;
    std::unordered_set<const ffi::Object*> active;
    std::vector<Var> ordered;
    std::function<void(const Var&)> visit;
    std::function<void(ffi::AnyView)> visit_type = [&](ffi::AnyView value) {
      const ffi::Object* object = value.as<ffi::Object>();
      if (!object) return;
      if (auto var = value.as<VarNode>()) {
        auto it = free_by_object.find(var);
        if (it == free_by_object.end() && !defined.count(var)) {
          Var dependency = ffi::GetRef<Var>(var);
          VarGetOrAllocId(dependency, true);
          it = free_by_object.emplace(var, dependency).first;
        }
        if (it != free_by_object.end()) visit(it->second);
        return;
      }
      if (!active.insert(object).second) return;
      if (auto array = value.as<ffi::Array<ffi::Any>>()) {
        for (const ffi::Any& child : *array) visit_type(child);
      } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
        for (const auto& [key, child] : *map) {
          visit_type(key);
          visit_type(child);
        }
      } else {
        ffi::reflection::ForEachFieldInfo(TVMFFIGetTypeInfo(object->type_index()),
                                          [&](const TVMFFIFieldInfo* field) {
                                            visit_type(ffi::reflection::FieldGetter(field)(object));
                                          });
      }
      active.erase(object);
    };
    visit = [&](const Var& var) {
      if (!ordered_objects.insert(var.get()).second) return;
      visit_type(var->ty);
      ordered.push_back(var);
    };
    // The index may encounter shape variables through a cached type before
    // it reaches the value that uses that type.  Start with non-scalar roots,
    // then emit any independent scalar roots in their original order.
    for (const Var& var : free_vars) {
      if (!var->ty.as<PrimType>()) visit(var);
    }
    for (const Var& var : free_vars) {
      if (var->ty.as<PrimType>()) visit(var);
    }
    for (const Var& var : ordered) {
      IdDoc id = IdDoc(VarGetOrAllocId(var, true)->name);
      RecordOrigin(id, var);
      if (auto ty = var->ty.as<PrimType>()) {
        ExprDoc creation = NamespaceDoc("ir")->Attr("dynamic")->Call(
            {LiteralDoc::Str(var->name, std::nullopt)}, {"dtype"},
            {LiteralDoc::DataType(ty.value()->dtype, std::nullopt)});
        Emit(AssignDoc(id, creation, std::nullopt), var);
      } else {
        TVM_FFI_CHECK(!var->ty.IsMissing(), TypeError)
            << "printer free non-scalar variable requires a type annotation";
        ExprDoc annotation = Translate(var->ty).value();
        ffi::Optional<ExprDoc> initializer = std::nullopt;
        if (var->ty.as<PointerTypeNode>()) {
          // An outer Python annotation does not bind the free pointer variable.
          initializer = annotation.as<CallDoc>() ? annotation : annotation->Call({});
        }
        Emit(AssignDoc(id, initializer, annotation), var);
      }
    }
  }

  ffi::List<Doc> DeclarePendingVariables() {
    return WithDocScope([&]() {
      auto pending = GetImplicitDefs();
      std::function<void(const Var&)> declare = [&](const Var& var) {
        if (!pending.count(var)) return;
        IdDoc id = IdDoc(VarGetOrAllocId(var, true)->name);
        RecordOrigin(id, var);
        ffi::Optional<ExprDoc> rhs = std::nullopt;
        ffi::Optional<ExprDoc> annotation = std::nullopt;
        if (auto primitive = var->ty.as<PrimType>()) {
          rhs = NamespaceDoc("ir")->Attr("dynamic")->Call(
              {LiteralDoc::Str(var->name, std::nullopt)}, {"dtype"},
              {LiteralDoc::DataType(primitive.value()->dtype, std::nullopt)});
        } else {
          TVM_FFI_CHECK(!var->ty.IsMissing(), TypeError)
              << "printer free non-scalar variable requires a type annotation";
          annotation = Translate(var->ty).value();
        }
        ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
            var->ty, [&](const Var& dependency) -> ffi::Expected<ffi::WalkResult> {
              declare(dependency);
              return ffi::WalkResult::Advance();
            });
        Emit(AssignDoc(id, rhs, annotation), var);
      };
      ffi::Dict<Var, IdDoc> candidates;
      for (const auto& [var, id] : pending) candidates.Set(var, id);
      std::vector<std::pair<Var, IdDoc>> ordered(candidates.begin(), candidates.end());
      std::sort(ordered.begin(), ordered.end(),
                [](const auto& a, const auto& b) { return a.second->name < b.second->name; });
      for (const auto& [var, id] : ordered) declare(var);
    });
  }

 private:
  void CollectObjects(ffi::AnyView value, std::unordered_set<const ffi::Object*>* ancestors) {
    const ffi::Object* object = value.as<ffi::Object>();
    if (!object || !ancestors->insert(object).second) return;
    ffi::ObjectRef ref = ffi::GetRef<ffi::ObjectRef>(object);
    objects_.push_back(ref);
    if (auto array = value.as<ffi::Array<ffi::Any>>()) {
      for (size_t i = 0; i < array->size(); ++i) CollectObjects((*array)[i], ancestors);
    } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      std::vector<std::pair<ffi::Any, ffi::Any>> entries(map->begin(), map->end());
      // Match the module parent's canonical member order when collecting free
      // declarations. Hash iteration must not choose their allocated names.
      if (std::all_of(entries.begin(), entries.end(),
                      [](const auto& item) { return item.first.template as<GlobalVarNode>(); })) {
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
          return a.first.template as<GlobalVarNode>()->name_hint <
                 b.first.template as<GlobalVarNode>()->name_hint;
        });
      }
      for (const auto& [key, child] : entries) CollectObjects(child, ancestors);
    } else {
      ffi::reflection::ForEachFieldInfo(
          TVMFFIGetTypeInfo(object->type_index()), [&](const TVMFFIFieldInfo* field) {
            CollectObjects(ffi::reflection::FieldGetter(field)(object), ancestors);
          });
    }
    ancestors->erase(object);
  }

  /*! \brief Original input retained throughout translation. */
  ffi::Any root_;
  /*! \brief Whether root signatures may use PEP 695 parameters. */
  bool use_pep695_{false};
  /*! \brief Borrowed identities local to a root Relax conditional. */
  std::unordered_set<const ffi::Object*> branch_local_;
  /*! \brief Original objects in deterministic occurrence order. */
  std::vector<ffi::ObjectRef> objects_;
};

details::TranslationResult TranslateComplete(ffi::AnyView obj, DocTranslationEngine* engine,
                                             ffi::Map<ffi::String, ffi::Any> extra_config) {
  auto binding_names = engine->GetExtraConfig<ffi::Array<ffi::String>>("script.binding_names", {});
  extra_config.Set("script.root_module", obj.as<IRModuleNode>() != nullptr);
  // Namespace names are reserved through the ordinary identifier allocator.
  engine->AllocId(engine->GetExtraConfig<ffi::String>("ir.prefix", "I"));
  engine->AllocId(engine->GetExtraConfig<ffi::String>("tirx.prefix", "T"));
  engine->AllocId(engine->GetExtraConfig<ffi::String>("s_tir.prefix", "Ts"));
  engine->AllocId(engine->GetExtraConfig<ffi::String>("relax.prefix", "R"));
  if (obj.as<IRModuleNode>()) {
    ffi::String module_name =
        binding_names.empty() ? engine->GetExtraConfig<ffi::String>("script.module_name", "Module")
                              : binding_names.back();
    extra_config.Set("script.module_name", module_name);
    engine->AllocId(module_name);
  }
  // Allocate generated names after all configured names. A dialect or module
  // named tvm/metadata must not shadow the loader or metadata registry.
  IdDoc metadata_id = engine->AllocId("metadata");
  IdDoc metadata_loader = engine->AllocId("tvm");
  extra_config.Set("script.metadata_name", metadata_id->name);
  extra_config.Set("script.use_pep695", engine->UsePEP695());
  extra_config.Set("script.syntax_sugar",
                   engine->GetExtraConfig<bool>("script.syntax_sugar", true));
  extra_config.Set("script.show_meta", engine->GetExtraConfig<bool>("script.show_meta", true));
  extra_config.Set("script.buffer_dtype",
                   ffi::DLDataTypeToString(ffi::StringToDLDataType(
                       engine->GetExtraConfig<ffi::String>("script.buffer_dtype", "float32"))));
  extra_config.Set("script.show_object_address",
                   engine->GetExtraConfig<bool>("script.show_object_address", false));
  engine->InitializeConfig(std::move(extra_config));
  auto docs = engine->WithDocScope([&]() {
    engine->DeclareFreeVariables();
    ffi::Optional<ExprDoc> expression = std::nullopt;
    if (auto type = obj.as<tirx::BufferType>()) {
      expression = details::TypeValue(engine, type.value(), false);
    } else {
      expression = engine->Translate(obj);
    }
    if (expression.has_value()) {
      if (!engine->GetExtraConfig<bool>("script.verbose_expr", true))
        engine->CurrentScopeDocs().clear();
      engine->Emit(ExprStmtDoc(expression.value()), obj.as<ffi::ObjectRef>());
    }
    if (!binding_names.empty() && !engine->CurrentScopeDocs().empty()) {
      Doc root = engine->CurrentScopeDocs().back();
      ffi::String name = binding_names.back();
      if (auto function = root.as<FunctionDoc>()) {
        if (auto func = obj.as<tirx::PrimFunc>()) {
          if (auto symbol = func.value()->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
            TVM_FFI_CHECK(symbol.value() == name, TypeError)
                << "printer PrimFunc global_symbol must match its definition name";
          }
        }
        function.value()->name = IdDoc(name);
      } else if (auto module = root.as<ClassDoc>()) {
        module.value()->name = IdDoc(name);
      }
    }
  });
  auto root_declarations = engine->DeclarePendingVariables();
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
  auto engine = ffi::make_object<DocTranslationEngine>(ir, extra_config);
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
