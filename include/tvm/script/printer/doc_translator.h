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
#ifndef TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_H_
#define TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_H_

#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/expr.h>
#include <tvm/script/printer/doc.h>

#include <optional>
#include <utility>

namespace tvm {

/*!
 * \brief Print an IR object as TVMScript, using repr when no translation hook exists.
 * \param node The input IR object.
 * \param config Optional translation and rendering configuration.
 * \return The rendered script or fallback representation.
 */
TVM_DLL std::string Script(const ffi::ObjectRef& node,
                           const ffi::Optional<PrinterConfig>& config = std::nullopt);

namespace script {
namespace printer {

class DocTranslatorObj;

/*!
 * \brief Type attribute containing a native or packed translation hook.
 */
inline constexpr const char* kDocTranslate = "__tvm_doc_translate__";
/*!
 * \brief Op-specific Call translation hook with the FDocTranslate contract.
 */
inline constexpr const char* kOpCallTranslate = "script_printer.call_translate";
/*!
 * \brief Source-Type hook receiving the original TensorLoad and optional binder.
 */
inline constexpr const char* kTensorLoadDocTranslate = "__tvm_doc_translate_tensor_load__";

/*!
 * \brief Engine callbacks borrowed for the translator's lifetime.
 */
struct DocTranslatorVTable {
  /*!
   * \brief Translate a value with an optional pre-mapped destination.
   * \param translator The borrowed translation context.
   * \param value The borrowed input value.
   * \param destination A borrowed Var, or nullptr when no binding is requested.
   * \return An owning Expected<Optional<ExprDoc>> encoded as TVMFFIAny. None means
   * emission completed the request; errors use Expected's error encoding.
   */
  TVMFFIAny (*translate)(DocTranslatorObj* translator, ffi::AnyView value,
                         const ffi::Object* destination) noexcept;
  /*!
   * \brief Reserve an identifier without mapping a variable.
   * \param translator The borrowed translation context.
   * \param hint The borrowed string name hint.
   * \return An owning Expected<IdDoc> encoded as TVMFFIAny, including errors.
   */
  TVMFFIAny (*alloc_id)(DocTranslatorObj* translator, ffi::AnyView hint) noexcept;
  /*!
   * \brief Find or allocate a variable's identifier.
   * \param translator The borrowed translation context.
   * \param var The borrowed variable.
   * \param explicit_def Whether to remove the binding from pending definitions.
   * \return An owning Expected<IdDoc> encoded as TVMFFIAny, including errors.
   */
  TVMFFIAny (*var_get_or_alloc_id)(DocTranslatorObj* translator, ffi::AnyView var,
                                   bool explicit_def) noexcept;
  /*!
   * \brief Get the current variable scope's pending definitions.
   * \param translator The borrowed translation context.
   * \return An owning Expected<Dict<Var, IdDoc>> encoded as TVMFFIAny, including errors.
   */
  TVMFFIAny (*get_implicit_defs)(DocTranslatorObj* translator) noexcept;
  /*!
   * \brief Begin a lexical variable scope.
   * \param translator The borrowed translation context.
   * \return Expected<void> encoded as TVMFFIAny, including errors.
   */
  TVMFFIAny (*begin_var_scope)(DocTranslatorObj* translator) noexcept;
  /*!
   * \brief End the current lexical variable scope.
   * \param translator The borrowed translation context.
   * \return Expected<void> encoded as TVMFFIAny, including errors.
   */
  TVMFFIAny (*end_var_scope)(DocTranslatorObj* translator) noexcept;
};

/*!
 * \brief Native translation hook with borrowed inputs and an owning result.
 * \param translator Translation context borrowed for the duration of the call.
 * \param value Input value borrowed for the duration of the call.
 * \param destination Optional binding variable borrowed for the duration of the call.
 * \return An owning expression Doc, or None after completing emission.
 * Return an expression for the caller to consume, or None after completing emission
 * and any requested binding. Packed hooks use the same contract and None for a null
 * destination; native errors cross the ABI through Expected.
 */
using FDocTranslate = ffi::reflection::NativeFunctionView<ffi::Optional<ExprDoc>(
    DocTranslatorObj* translator, ffi::AnyView value, const ffi::Object* destination)>;

/*!
 * \brief Eager translation context shared by registered hooks.
 * Hooks construct Docs, use explicit lexical scopes, and attach origins when emitting.
 *
 * \code{.cpp}
 * ffi::Optional<ExprDoc> Hook(DocTranslatorObj* d, ffi::AnyView value,
 *                             const ffi::Object* destination) {
 *   ExprDoc rhs = LiteralDoc::Str("value", std::nullopt);
 *   if (!destination) return rhs;
 *   auto var = ffi::GetRef<Var>(static_cast<const VarNode*>(destination));
 *   IdDoc lhs(d->VarGetOrAllocId(var, true)->name);
 *   d->Emit(AssignDoc(lhs, rhs, std::nullopt), var);
 *   return std::nullopt;
 * }
 * // Register with TypeAttrDef<Node>().attr(kDocTranslate, FDocTranslate::FromNative<&Hook>()).
 * \endcode
 */
class DocTranslatorObj : public ffi::Object {
 public:
  /*!
   * \brief Translate a value with an optional pre-mapped binding variable.
   * \param value The input value.
   * \param bind_var The binding variable, or None when no binding is requested.
   * \return An expression for the caller, or None when emission completed the request.
   * Callers requiring an expression use value(); for a specific Doc subtype,
   * use value().as_or_throw<T>().
   */
  ffi::Optional<ExprDoc> Translate(ffi::AnyView value,
                                   const ffi::Optional<Var> bind_var = std::nullopt) {
    auto result = ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Optional<ExprDoc>>(
                      vtable_->translate(this, value,
                                         bind_var.has_value() ? bind_var.value().get() : nullptr))
                      .value();
    if (result.has_value()) {
      if (const auto* origin = value.as<ffi::Object>()) {
        RecordOrigin(result.value(), ffi::GetRef<ffi::ObjectRef>(origin));
      }
    }
    return result;
  }
  /*!
   * \brief Append a completed Doc and record its optional origin.
   * \param doc The emitted Doc.
   * \param origin The original IR object represented by this emission.
   */
  void Emit(const Doc& doc, ffi::Optional<ffi::ObjectRef> origin = std::nullopt) {
    CurrentScopeDocs().push_back(doc);
    if (origin.has_value()) RecordOrigin(doc, origin.value());
  }

  /*!
   * \brief Reuse or allocate a variable's stable identifier.
   * \param var The variable to name.
   * \param explicit_def Whether this occurrence defines the variable.
   * \return The binding's IdDoc, with explicit definitions removed from pending entries.
   */
  IdDoc VarGetOrAllocId(const Var& var, bool explicit_def) {
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<IdDoc>(
               vtable_->var_get_or_alloc_id(this, var, explicit_def))
        .value();
  }
  /*!
   * \brief Reserve a fresh identifier without mapping a variable.
   * \param hint The preferred name.
   * \return The allocated identifier.
   */
  IdDoc AllocId(const ffi::String& hint) {
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<IdDoc>(vtable_->alloc_id(this, hint))
        .value();
  }
  /*!
   * \brief Get the current variable scope's pending definitions.
   * \return The live mutable dictionary of variables and their identifiers.
   */
  ffi::Dict<Var, IdDoc> GetImplicitDefs() {
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Dict<Var, IdDoc>>(
               vtable_->get_implicit_defs(this))
        .value();
  }

  /*!
   * \brief Read a typed extension option.
   * \tparam T The option type.
   * \param name The option name.
   * \param fallback The value to use when absent.
   * \return The typed option or fallback, rejecting an existing wrong type.
   */
  template <typename T>
  T GetExtraConfig(const ffi::String& name, T fallback) const {
    auto it = extra_config_.find(name);
    return it == extra_config_.end() ? fallback : (*it).second.as_or_throw<T>();
  }
  /*!
   * \brief Get or initialize typed extension state.
   * \tparam T The state type, default constructed when absent.
   * \param name The state name.
   * \return The stored value, rejecting an existing wrong type; mutable handles share storage.
   */
  template <typename T>
  T GetOrCreateExtraState(const ffi::String& name) {
    auto it = extra_state_.find(name);
    if (it != extra_state_.end()) return (*it).second.as_or_throw<T>();
    T value{};
    extra_state_.Set(name, value);
    return value;
  }
  /*!
   * \brief Replace or remove extension state.
   * \param name The state name.
   * \param value The replacement value, or nullopt to remove the entry.
   * \return The previous value, or nullopt when absent.
   */
  std::optional<ffi::Any> ExchangeExtraState(const ffi::String& name,
                                             std::optional<ffi::Any> value) {
    auto previous = extra_state_.Get(name);
    if (value.has_value()) {
      extra_state_.Set(name, std::move(value.value()));
    } else {
      extra_state_.erase(name);
    }
    return previous;
  }

  /*!
   * \brief Begin a lexical variable scope independent of Doc collection.
   */
  void BeginVarScope() {
    ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<void>(vtable_->begin_var_scope(this)).value();
  }
  /*!
   * \brief End the latest lexical variable scope, retaining the root scope.
   */
  void EndVarScope() {
    ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<void>(vtable_->end_var_scope(this)).value();
  }
  /*!
   * \brief Push an empty Doc collection.
   */
  void BeginDocScope() { doc_scopes_.push_back(ffi::List<Doc>()); }
  /*!
   * \brief Pop the latest Doc collection.
   * \return The collected Docs.
   */
  ffi::List<Doc> EndDocScope() {
    auto docs = CurrentScopeDocs();
    doc_scopes_.pop_back();
    return docs;
  }
  /*!
   * \brief Get the current Doc collection without popping it.
   * \return A mutable handle sharing the active collection's storage.
   */
  ffi::List<Doc> CurrentScopeDocs() const {
    TVM_FFI_CHECK(!doc_scopes_.empty(), ValueError)
        << "printer emission requires a collection scope";
    return doc_scopes_.back();
  }
  /*!
   * \brief Run a callback in a balanced Doc collection scope.
   * \tparam Callback The callable type.
   * \param callback The callback that emits Docs.
   * \return The collected Docs.
   */
  template <typename Callback>
  ffi::List<Doc> WithDocScope(Callback&& callback) {
    size_t depth = doc_scopes_.size();
    BeginDocScope();
    try {
      std::forward<Callback>(callback)();
      TVM_FFI_CHECK(doc_scopes_.size() == depth + 1, ValueError)
          << "printer unbalanced Doc collection scopes";
      return EndDocScope();
    } catch (...) {
      while (doc_scopes_.size() > depth) doc_scopes_.pop_back();
      throw;
    }
  }

  /*!
   * \brief Get the owned Doc-to-original-object associations.
   * \return A mutable handle sharing the origin dictionary's storage.
   */
  ffi::Dict<Doc, ffi::ObjectRef> GetDocOrigins() const { return doc_origins_; }
  /*!
   * \brief Set a Doc's original IR object, replacing any previous association.
   * \param doc The constructed Doc.
   * \param origin The original IR object.
   */
  void RecordOrigin(const Doc& doc, const ffi::ObjectRef& origin) { doc_origins_.Set(doc, origin); }
  /*!
   * \brief Dispatch directly to the registered native or packed hook.
   * \param value The input value.
   * \param destination A borrowed Var, or nullptr when no binding is requested.
   * \return An expression for the caller, or None after completed emission.
   */
  ffi::Optional<ExprDoc> DefaultTranslate(ffi::AnyView value,
                                          const ffi::Object* destination = nullptr) {
    static ffi::reflection::TypeAttrColumn column(kDocTranslate);
    ffi::AnyView attr = column[value.type_index()];
    if (attr.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
      return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Optional<ExprDoc>>(
                 reinterpret_cast<decltype(DocTranslatorVTable::translate)>(attr.cast<void*>())(
                     this, value, destination))
          .value();
    }
    if (attr.type_index() == ffi::TypeIndex::kTVMFFIFunction) {
      ffi::Any destination_arg = nullptr;
      if (destination) destination_arg = ffi::GetRef<ffi::ObjectRef>(destination);
      return attr.cast<ffi::Function>()
          .CallExpected<ffi::Optional<ExprDoc>>(this, value, destination_arg)
          .value();
    }
    TVM_FFI_THROW(TypeError) << (attr.type_index() == ffi::TypeIndex::kTVMFFINone
                                     ? std::string("printer has no Doc hook for ") +
                                           value.GetTypeKey()
                                     : "printer Doc hook must be a native pointer or ffi.Function");
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("script.printer.DocTranslator", DocTranslatorObj, ffi::Object);

 protected:
  /*!
   * \brief Initialize an engine's translation context.
   * \param vtable The engine callbacks, valid for this object's lifetime.
   * \param extra_config The owned configuration snapshot.
   */
  explicit DocTranslatorObj(const DocTranslatorVTable* vtable,
                            ffi::Map<ffi::String, ffi::Any> extra_config = {})
      : extra_config_(std::move(extra_config)), vtable_(vtable) {}

  /*!
   * \brief Finish the engine's configuration before translating any values.
   * \param extra_config The completed owned configuration snapshot.
   */
  void InitializeExtraConfig(ffi::Map<ffi::String, ffi::Any> extra_config) {
    extra_config_ = std::move(extra_config);
  }

 private:
  /*!
   * \brief Owned extension configuration.
   * Options are read eagerly by translation hooks.
   */
  ffi::Map<ffi::String, ffi::Any> extra_config_;
  /*!
   * \brief Owned mutable extension state.
   * Values retain their objects and mutable handles share storage.
   */
  ffi::Dict<ffi::String, ffi::Any> extra_state_;
  /*!
   * \brief Active Doc collections, independent of lexical variable scopes.
   */
  ffi::List<ffi::List<Doc>> doc_scopes_;
  /*!
   * \brief Owned associations between Docs and their original IR objects.
   */
  ffi::Dict<Doc, ffi::ObjectRef> doc_origins_;
  /*!
   * \brief Borrowed engine callbacks valid for this object's lifetime.
   */
  const DocTranslatorVTable* vtable_;
};

/*!
 * \brief Convert collected Docs to statements in their original order.
 * \param docs The Docs to convert, without modifying the collection.
 * \return An owning array containing each StmtBlockDoc's statements in place,
 *         and each other Doc converted with as_or_throw<StmtDoc>().
 * \throws ffi::Error If a Doc is neither a StmtBlockDoc nor a StmtDoc.
 */
inline ffi::Array<StmtDoc> ToStmtDocArray(const ffi::List<Doc>& docs) {
  ffi::Array<StmtDoc> statements;
  for (const auto& doc : docs) {
    if (auto block = doc.as<StmtBlockDoc>()) {
      for (const auto& stmt : block.value()->stmts) statements.push_back(stmt);
    } else {
      statements.push_back(doc.as_or_throw<StmtDoc>());
    }
  }
  return statements;
}

/*!
 * \brief Owning reference to a hook translation context.
 */
class DocTranslator : public ffi::ObjectRef {
 public:
  /*!
   * \brief Retain a translation context.
   * \param n The context object.
   */
  explicit DocTranslator(ffi::ObjectPtr<DocTranslatorObj> n) : ObjectRef(std::move(n)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DocTranslator, ffi::ObjectRef, DocTranslatorObj);
};

/*!
 * \brief Translate IR into a complete Doc including declarations and metadata setup.
 * \param ir The input IR value.
 * \param doc_origins Optional output replaced with the owned origins only on success.
 * \param extra_config The translation configuration snapshot.
 * \return The translated Doc, with text headers and diagnostic recovery left to the text adapter.
 */
TVM_DLL Doc DocTranslate(ffi::AnyView ir, ffi::Dict<Doc, ffi::ObjectRef>* doc_origins = nullptr,
                         ffi::Map<ffi::String, ffi::Any> extra_config = {});
/*!
 * \brief Translate IR, recover diagnostic paths, and render Python text.
 * \param obj The input IR object.
 * \param config The translation and rendering options.
 * \return The rendered script.
 */
TVM_DLL ffi::String Script(const ffi::ObjectRef& obj, const PrinterConfig& config);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_DOC_TRANSLATOR_H_
