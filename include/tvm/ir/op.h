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

/*!
 * \file tvm/ir/op.h
 * \brief Canonical primitive operators and their registration.
 */
#ifndef TVM_IR_OP_H_
#define TVM_IR_OP_H_

#include <tvm/ffi/container/list.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/native_function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/type.h>

#include <string>
#include <type_traits>
#include <utility>

namespace tvm {

template <typename>
class OpAttrMap;

/*!
 * \brief An operator's result type independent of operands, attributes, and type arguments.
 * When present, this concrete type takes precedence over FInferType.
 */
using TFixedReturnType = Type;

/*! \brief Infer a Call's result type from its explicit inputs without builder state. */
using FInferType = ffi::reflection::NativeFunctionView<Type(const CallNode* call)>;

/*!
 * \brief The fully qualified TVMScript name, including its dialect namespace.
 */
using TScriptPrinterName = ffi::String;

/*! \brief An operator argument's name and documentation. */
class ArgumentInfoNode : public ffi::Object {
 public:
  /*! \brief Argument name. */
  ffi::String name;
  /*! \brief Argument documentation. */
  ffi::String doc;

  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<ArgumentInfoNode>()
        .def_ro("name", &ArgumentInfoNode::name)
        .def_ro("doc", &ArgumentInfoNode::doc);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.ArgumentInfo", ArgumentInfoNode, ffi::Object);
};

/*! \brief Managed reference to an argument descriptor. */
class ArgumentInfo : public ffi::ObjectRef {
 public:
  explicit ArgumentInfo(ffi::ObjectPtr<ArgumentInfoNode> n) : ffi::ObjectRef(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(n != nullptr);
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ArgumentInfo, ffi::ObjectRef, ArgumentInfoNode);
};

/*! \brief Metadata for a canonical primitive operator invoked through Call. */
class OpNode : public ExprNode {
 private:
  // Dense process-local index into attribute columns; never serialized.
  uint32_t index_{0};

 public:
  /*! \brief Canonical operator name. */
  ffi::String name;
  /*! \brief Operator documentation. */
  ffi::String doc;
  /*! \brief Descriptors for the ordered required value-argument prefix. */
  ffi::Array<ArgumentInfo> args_info;
  /*! \brief Descriptors for the ordered required type-argument prefix. */
  ffi::Array<ArgumentInfo> ty_args_info;
  /*! \brief Named variadic value tail, excluded from the required args_info prefix. */
  ffi::Optional<ArgumentInfo> var_args_info;
  /*! \brief Named variadic type-argument tail, excluded from ty_args_info. */
  ffi::Optional<ArgumentInfo> var_ty_args_info;
  /*! \brief Attribute object type key, or empty when no attrs type is declared. */
  ffi::String attrs_type_key;
  /*! \brief Runtime index corresponding to attrs_type_key; not serialized. */
  uint32_t attrs_type_index{0};
  /*! \brief Process-local validation callback, omitted from reflection. */
  ffi::Any validator;
  /*! \brief Whether the validator was explicitly registered rather than generated. */
  bool validator_is_custom{false};

  TVM_DLL static void RegisterReflection();

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.Op", OpNode, ExprNode);

 private:
  friend class OpRegistry;
  friend class OpDef;
  friend class Op;
  template <typename>
  friend class OpAttrMap;
};

/*!
 * \brief Reference-counted handle to a canonical named operator.
 *
 * OpDef temporarily builds metadata on the same Op returned by Get. Independent
 * registrations may attach different attributes; replacing an existing attribute
 * requires explicit override. Cached attribute maps observe subsequent changes.
 * args_info describes required value operands; var_args_info permits a suffix.
 * ty_args_info describes type arguments; a typed signature installs a
 * validator that checks their count and classes.
 *
 * \code
 * TVM_FFI_STATIC_INIT_BLOCK() {
 *   OpDef("example.identity", "Return the input expression.")
 *       .signature(sig::arg("value", "The input expression."))
 *       .set_attr<bool>("FPurity", true);
 * }
 * // Copies the handle; the canonical node is shared.
 * Op op = Op::Get("example.identity");
 * bool pure = Op::GetAttrMap<bool>("FPurity")[op];
 * \endcode
 */
class Op : public Expr {
 public:
  /*! \brief Construct a handle from a defined node. \param n The operator node. */
  explicit Op(ffi::ObjectPtr<OpNode> n) : Expr(std::move(n)) {
    TVM_FFI_CHECK(defined(), ValueError) << "Op expects a defined OpNode";
  }

  /*!
   * \brief Get a typed live view of a registered attribute column.
   * \tparam ValueType The attribute value type.
   * \param attr_name The column name.
   * \return A shared view that observes registration, replacement, reset, and growth.
   * \throws InternalError if the column has never been registered.
   */
  template <typename ValueType>
  static OpAttrMap<ValueType> GetAttrMap(const ffi::String& attr_name);
  /*!
   * \brief Check whether an attribute column exists, even if some Ops lack a value.
   * \param attr_name The column name.
   * \return Whether the column has been registered.
   */
  TVM_DLL static bool HasAttrMap(const ffi::String& attr_name);
  /*!
   * \brief Look up a registered operator by name.
   * \param op_name The canonical name.
   * \return A copy of the reference-counted handle to the canonical node.
   * \throws AttributeError if the operator is not registered.
   */
  TVM_DLL static Op Get(const ffi::String& op_name);
  /*! \brief List registered operator names. \return Names in unspecified order. */
  TVM_DLL static ffi::Array<ffi::String> ListNames();

  /*! \brief Validate a Call with this Op's callback, when one is registered. */
  TVM_FFI_INLINE void Validate(const CallNode* call) const {
    const auto& validator = get()->validator;
    if (validator != nullptr) {
      if (TVM_FFI_PREDICT_FALSE(!call)) {
        ThrowInvalidCall(get());
      }
      using View = ffi::reflection::NativeFunctionView<void(const CallNode*)>;
      // set_validator stores only an owning NativeFunction with this signature.
      ffi::details::AnyUnsafe::CopyFromAnyViewAfterCheck<View>(validator)
          .CallExpected(call)
          .value();
    }
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Op, Expr, OpNode);

 private:
  TVM_DLL static void ThrowInvalidCall(const OpNode* op);
  TVM_DLL static ffi::List<ffi::Any> GetAttrColumn(const ffi::String& attr_name);
};

namespace sig {

/*! \brief Base counts for descriptors accepted by OpDef::signature. */
struct SignatureTrait {
  /*! \brief Number of required value arguments contributed by this descriptor. */
  static constexpr size_t kArgsCount = 0;
  /*! \brief Number of required type arguments contributed by this descriptor. */
  static constexpr size_t kTyArgsCount = 0;
  /*! \brief Number of variadic value tails contributed by this descriptor. */
  static constexpr size_t kVarArgsCount = 0;
  /*! \brief Number of variadic type tails contributed by this descriptor. */
  static constexpr size_t kVarTyArgsCount = 0;
  /*! \brief Number of Call.attrs descriptors contributed by this descriptor. */
  static constexpr size_t kCallAttrsCount = 0;
};

/*! \brief Shared inputs and indexes for a typed signature validation fold. */
struct ValidateState {
  /*! \brief Call being validated. */
  const CallNode* call;
  /*! \brief Canonical operator and its current argument metadata. */
  const OpNode* op;
  /*! \brief Index of the next value argument after the upfront arity check. */
  size_t args_index = 0;
  /*! \brief Index of the next type argument after the upfront arity check. */
  size_t ty_args_index = 0;
};

namespace details {

template <typename T>
inline bool ReportTypeMismatch(ffi::Expected<void>* out, const ValidateState* state, bool type_arg,
                               size_t index, ffi::AnyView actual) {
  const auto& fixed = type_arg ? state->op->ty_args_info : state->op->args_info;
  const auto& tail = type_arg ? state->op->var_ty_args_info : state->op->var_args_info;
  std::string name;
  if (index < fixed.size()) {
    name = fixed[index]->name;
  } else if (tail.has_value()) {
    name = tail.value()->name;
  }
  std::string name_clause = name.empty() ? "" : " (`" + name + "`)";
  TVMFFIAny raw = actual.CopyToTVMFFIAny();
  *out = TVM_FFI_UNEXPECTED(TypeError)
         << "Op `" << state->op->name << "`: `" << (type_arg ? "Call.ty_args[" : "Call.args[")
         << index << "]`" << name_clause << " expected `" << ffi::TypeTraits<T>::TypeStr()
         << "`, got `" << ffi::TypeTraits<T>::GetMismatchTypeInfo(&raw) << "`.";
  return false;
}

}  // namespace details

/*! \brief One value argument of type T, for example `sig::arg<PrimExpr>("index")`
 * for a required primitive index expression. */
template <typename T = Expr>
struct arg : SignatureTrait {
  static_assert(std::is_base_of_v<Expr, T>, "arg<T> expects an expression view");
  /*! \brief Contributes one required value argument. */
  static constexpr size_t kArgsCount = 1;
  /*! \brief Argument name recorded in operator metadata. */
  ffi::String name;
  /*! \brief Argument documentation recorded in operator metadata. */
  ffi::String doc;

  explicit arg(ffi::String name, ffi::String doc = "")
      : name(std::move(name)), doc(std::move(doc)) {}

  void CollectMetadata(OpNode* op) const {
    auto info = ffi::make_object<ArgumentInfoNode>();
    info->name = name;
    info->doc = doc;
    op->args_info.push_back(ArgumentInfo(std::move(info)));
  }

  static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
    if constexpr (!std::is_same_v<T, Expr>) {
      ffi::AnyView value(state->call->args.GetArrayObj()->begin()[state->args_index]);
      if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
        return details::ReportTypeMismatch<T>(out, state, false, state->args_index, value);
      }
    }
    ++state->args_index;
    return true;
  }
};

/*! \brief Zero or more trailing value arguments of type T, for example
 * `sig::var_args<PrimExpr>("indices")` for primitive index operands. */
template <typename T = Expr>
struct var_args : SignatureTrait {
  static_assert(std::is_base_of_v<Expr, T>, "var_args<T> expects an expression view");
  /*! \brief Contributes one variadic value tail. */
  static constexpr size_t kVarArgsCount = 1;
  /*! \brief Tail name recorded in operator metadata. */
  ffi::String name;
  /*! \brief Tail documentation recorded in operator metadata. */
  ffi::String doc;

  explicit var_args(ffi::String name, ffi::String doc = "")
      : name(std::move(name)), doc(std::move(doc)) {}

  void CollectMetadata(OpNode* op) const {
    auto info = ffi::make_object<ArgumentInfoNode>();
    info->name = name;
    info->doc = doc;
    op->var_args_info = ArgumentInfo(std::move(info));
  }

  static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
    if constexpr (!std::is_same_v<T, Expr>) {
      const ffi::Any* values = state->call->args.GetArrayObj()->begin();
      for (; state->args_index != state->call->args.size(); ++state->args_index) {
        ffi::AnyView value(values[state->args_index]);
        if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                  !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
          return details::ReportTypeMismatch<T>(out, state, false, state->args_index, value);
        }
      }
    } else {
      state->args_index = state->call->args.size();
    }
    return true;
  }
};

/*! \brief One type argument of type T, for example `sig::ty_arg<PrimType>("dtype")`
 * for a required primitive type. */
template <typename T = Type>
struct ty_arg : SignatureTrait {
  static_assert(std::is_base_of_v<Type, T>, "ty_arg<T> expects a type view");
  /*! \brief Contributes one required type argument. */
  static constexpr size_t kTyArgsCount = 1;
  /*! \brief Type-argument name recorded in operator metadata. */
  ffi::String name;
  /*! \brief Type-argument documentation recorded in operator metadata. */
  ffi::String doc;

  explicit ty_arg(ffi::String name, ffi::String doc = "")
      : name(std::move(name)), doc(std::move(doc)) {}

  void CollectMetadata(OpNode* op) const {
    auto info = ffi::make_object<ArgumentInfoNode>();
    info->name = name;
    info->doc = doc;
    op->ty_args_info.push_back(ArgumentInfo(std::move(info)));
  }

  static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
    if constexpr (!std::is_same_v<T, Type>) {
      ffi::AnyView value(state->call->ty_args.GetArrayObj()->begin()[state->ty_args_index]);
      if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
        return details::ReportTypeMismatch<T>(out, state, true, state->ty_args_index, value);
      }
    }
    ++state->ty_args_index;
    return true;
  }
};

/*! \brief Zero or more trailing type arguments of type T, for example
 * `sig::var_ty_args<PrimType>("dtypes")` for optional primitive types. */
template <typename T = Type>
struct var_ty_args : SignatureTrait {
  static_assert(std::is_base_of_v<Type, T>, "var_ty_args<T> expects a type view");
  /*! \brief Contributes one variadic type tail. */
  static constexpr size_t kVarTyArgsCount = 1;
  /*! \brief Tail name recorded in operator metadata. */
  ffi::String name;
  /*! \brief Tail documentation recorded in operator metadata. */
  ffi::String doc;

  explicit var_ty_args(ffi::String name, ffi::String doc = "")
      : name(std::move(name)), doc(std::move(doc)) {}

  void CollectMetadata(OpNode* op) const {
    auto info = ffi::make_object<ArgumentInfoNode>();
    info->name = name;
    info->doc = doc;
    op->var_ty_args_info = ArgumentInfo(std::move(info));
  }

  static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
    if constexpr (!std::is_same_v<T, Type>) {
      const ffi::Any* values = state->call->ty_args.GetArrayObj()->begin();
      for (; state->ty_args_index != state->call->ty_args.size(); ++state->ty_args_index) {
        ffi::AnyView value(values[state->ty_args_index]);
        if (TVM_FFI_PREDICT_FALSE(value == nullptr ||
                                  !ffi::details::AnyUnsafe::CheckAnyViewStrict<T>(value))) {
          return details::ReportTypeMismatch<T>(out, state, true, state->ty_args_index, value);
        }
      }
    } else {
      state->ty_args_index = state->call->ty_args.size();
    }
    return true;
  }
};

/*!
 * \brief Require Call.attrs to hold node type T and record its metadata.
 *
 * \code
 * OpDef("example.with_attrs").signature(sig::call_attrs<DictAttrsNode>());
 * // Call.attrs must contain a DictAttrsNode for this operator.
 * \endcode
 */
template <typename T>
struct call_attrs : SignatureTrait {
  static_assert(std::is_base_of_v<AttrsNode, T>, "call_attrs<T> expects an attribute node type");
  /*! \brief Contributes one required Call.attrs descriptor. */
  static constexpr size_t kCallAttrsCount = 1;

  void CollectMetadata(OpNode* op) const {
    uint32_t index = T::RuntimeTypeIndex();
    op->attrs_type_key = T::_type_key;
    op->attrs_type_index = index;
  }

  static bool Validate(ValidateState* state, ffi::Expected<void>* out) {
    ffi::AnyView attrs(state->call->attrs);
    if (TVM_FFI_PREDICT_FALSE(!state->call->attrs.defined() ||
                              !ffi::details::AnyUnsafe::CheckAnyViewStrict<const T*>(attrs))) {
      TVMFFIAny raw = attrs.CopyToTVMFFIAny();
      std::string actual = state->call->attrs.defined()
                               ? "`" + ffi::TypeTraits<const T*>::GetMismatchTypeInfo(&raw) + "`"
                               : "None";
      *out = TVM_FFI_UNEXPECTED(TypeError)
             << "Op `" << state->op->name << "`: Call.attrs expected `"
             << ffi::TypeTraits<const T*>::TypeStr() << "`, got " << actual;
      return false;
    }
    return true;
  }
};

}  // namespace sig

/*! \brief Noncopyable temporary builder for a canonical Op; see Op for the example. */
class OpDef {
 public:
  /*!
   * \brief Get or create the named Op without changing its documentation.
   * \param name The canonical operator name.
   */
  TVM_DLL explicit OpDef(const ffi::String& name);
  /*!
   * \brief Get or create the named Op and set its documentation.
   * \param name The canonical operator name.
   * \param doc Operator documentation, replacing any existing documentation.
   */
  TVM_DLL OpDef(const ffi::String& name, const ffi::String& doc);
  OpDef(const OpDef&) = delete;
  OpDef& operator=(const OpDef&) = delete;
  OpDef(OpDef&&) = delete;
  OpDef& operator=(OpDef&&) = delete;
  /*! \brief Get the canonical operator. \return A copied handle, safe after this builder dies. */
  Op op() const { return op_; }
  /*!
   * \brief Append a required value-argument descriptor.
   * \param name Argument name.
   * \param doc Argument documentation.
   * \return This builder. Without a variadic descriptor, this is part of the complete list.
   */
  TVM_DLL OpDef& add_arg(const ffi::String& name, const ffi::String& doc);
  /*!
   * \brief Set the attribute object type key and runtime index together.
   * \tparam AttrsType The attribute object node type.
   * \return This builder.
   */
  template <typename AttrsType>
  OpDef& attrs_type() {
    uint32_t index = AttrsType::RuntimeTypeIndex();
    get()->attrs_type_key = AttrsType::_type_key;
    get()->attrs_type_index = index;
    return *this;
  }
  /*!
   * \brief Resolve and set the attribute object type key and runtime index together.
   * \param key A registered attribute object type key.
   * \return This builder. An unknown key raises an error before modifying metadata.
   */
  TVM_DLL OpDef& set_attrs_type_key(const ffi::String& key);
  /*!
   * \brief Append a descriptor for a type argument in Call.ty_args.
   * \param name Type-argument name.
   * \param doc Type-argument documentation.
   * \return This builder. Inference retains responsibility for type-argument counts.
   */
  TVM_DLL OpDef& add_ty_arg(const ffi::String& name, const ffi::String& doc);
  /*!
   * \brief Register a complete typed Call signature and its validator.
   *
   * Fixed descriptors require exactly one value/type argument each. A variadic
   * tail accepts zero or more trailing arguments of its category. Validation
   * throws on arity or class mismatch; it does not infer types or mutate the
   * Call. Call::Validate invokes the validator separately from construction. Without call_attrs<T>,
   * Call.attrs is unconstrained. Repeated registration replaces metadata but retains any existing
   * validator, including a generated one. Replace the callback explicitly if its executable checks
   * must change.
   *
   * \code
   * OpDef("example.op")
   *     .signature(
   *         sig::arg("value"),
   *         sig::arg<PrimExpr>("index"),
   *         sig::var_args<PrimExpr>("rest"),
   *         sig::ty_arg("T"),
   *         sig::var_ty_args("Ts"),
   *         sig::call_attrs<DictAttrsNode>());
   * \endcode
   */
  template <typename... Specs>
  OpDef& signature(const Specs&... specs) {
    auto updated = ffi::make_object<OpNode>();
    (ApplySignatureTrait(updated.get(), specs), ...);
    if (op_->validator == nullptr) {
      using View = ffi::reflection::NativeFunctionView<void(const CallNode*)>;
      set_validator(View::FromNative<&ValidateSignature<Specs...>>());
      get()->validator_is_custom = false;
    }
    OpNode* target = get();
    target->args_info = std::move(updated->args_info);
    target->ty_args_info = std::move(updated->ty_args_info);
    target->var_args_info = std::move(updated->var_args_info);
    target->var_ty_args_info = std::move(updated->var_ty_args_info);
    target->attrs_type_key = std::move(updated->attrs_type_key);
    target->attrs_type_index = updated->attrs_type_index;
    return *this;
  }
  /*!
   * \brief Register an extensible attribute, rejecting duplicates unless overridden.
   * \tparam ValueType The attribute value type.
   * \param attr_name Attribute column name.
   * \param value Non-null attribute value.
   * \param override Whether to replace an existing value; no prior value is retained.
   * \return This builder. Existing cached maps observe the new value.
   */
  template <typename ValueType>
  OpDef& set_attr(const ffi::String& attr_name, const ValueType& value, bool override = false) {
    UpdateAttr(attr_name, ffi::Any(value), override);
    return *this;
  }
  /*!
   * \brief Register a validator that checks a Call with this operator.
   *
   * The callback accepts a `const CallNode*` and reports invalid input by
   * throwing or returning an `Expected<void>` error. A native function pointer
   * can be bound with NativeFunctionView::FromNative. A borrowed packed function
   * may also be passed while it remains alive for this call. The setter retains an owning
   * copy, so the original packed function may then be destroyed.
   * Call::Validate invokes it explicitly; Relax normalization and
   * well-formedness also validate when inputs are ready.
   * A signature installs its generated validator only if none is registered,
   * so a custom validator registered first takes precedence. Replacing a
   * generated or custom validator requires override; duplicate registration
   * without override raises ValueError. NativeFunctionView cannot be null.
   *
   * \param validator Callback to install.
   * \param override Whether to replace the current validator.
   * \return This builder.
   */
  OpDef& set_validator(ffi::reflection::NativeFunctionView<void(const CallNode*)> validator,
                       bool override = false) {
    TVM_FFI_CHECK(override || op_->validator == nullptr, ValueError)
        << "Validator of " << op_->name << " is already registered";
    get()->validator = ffi::reflection::NativeFunction<void(const CallNode*)>::From(validator);
    get()->validator_is_custom = true;
    return *this;
  }
  /*!
   * \brief Remove the current attribute value, with no priority fallback.
   * \param attr_name Attribute column name; missing columns and values are ignored.
   * \return This builder. Existing cached maps observe the removal.
   */
  TVM_DLL OpDef& reset_attr(const ffi::String& attr_name);

 private:
  template <typename Spec>
  static void ApplySignatureTrait(OpNode* op, const Spec& spec) {
    static_assert(std::is_base_of_v<sig::SignatureTrait, Spec>, "Unknown signature descriptor");
    spec.CollectMetadata(op);
  }

  template <typename... Specs>
  TVM_FFI_INLINE static ffi::Expected<void> ValidateSignature(const CallNode* call) noexcept {
    static_assert((std::is_base_of_v<sig::SignatureTrait, Specs> && ...),
                  "Unknown signature descriptor");
    constexpr size_t n_args = (size_t{0} + ... + Specs::kArgsCount);
    constexpr size_t n_ty_args = (size_t{0} + ... + Specs::kTyArgsCount);
    constexpr size_t n_var_args = (size_t{0} + ... + Specs::kVarArgsCount);
    constexpr size_t n_var_ty_args = (size_t{0} + ... + Specs::kVarTyArgsCount);
    constexpr size_t n_attrs = (size_t{0} + ... + Specs::kCallAttrsCount);
    static_assert(n_var_args <= 1 && n_var_ty_args <= 1 && n_attrs <= 1,
                  "Duplicate signature tail or call_attrs");
    const OpNode* op = call ? call->op.as<OpNode>() : nullptr;
    if (TVM_FFI_PREDICT_FALSE(!op)) {
      return TVM_FFI_UNEXPECTED(TypeError) << "Invalid Op Call";
    }
    if (TVM_FFI_PREDICT_FALSE(!call->args.defined() || !call->ty_args.defined())) {
      return TVM_FFI_UNEXPECTED(TypeError) << "Op `" << op->name << "`: invalid Call";
    }
    ffi::Expected<void> result;
    auto* out = &result;
    if (TVM_FFI_PREDICT_FALSE(n_var_args ? call->args.size() < n_args
                                         : call->args.size() != n_args)) {
      *out = TVM_FFI_UNEXPECTED(TypeError)
             << "Op `" << op->name << "`: Call.args expected " << (n_var_args ? "at least " : "")
             << n_args << (n_args == 1 ? " argument" : " arguments") << ", got "
             << call->args.size();
      return result;
    }
    if (TVM_FFI_PREDICT_FALSE(n_var_ty_args ? call->ty_args.size() < n_ty_args
                                            : call->ty_args.size() != n_ty_args)) {
      *out = TVM_FFI_UNEXPECTED(TypeError)
             << "Op `" << op->name << "`: Call.ty_args expected "
             << (n_var_ty_args ? "at least " : "") << n_ty_args
             << (n_ty_args == 1 ? " type argument" : " type arguments") << ", got "
             << call->ty_args.size();
      return result;
    }
    if constexpr (sizeof...(Specs) != 0) {
      sig::ValidateState state{call, op};
      if (TVM_FFI_PREDICT_FALSE(!(Specs::Validate(&state, out) && ...))) {
        return result;
      }
    }
    return result;
  }

  OpNode* get() { return const_cast<OpNode*>(op_.operator->()); }
  TVM_DLL void UpdateAttr(const ffi::String& attr_name, ffi::Any value, bool override);
  Op op_;
};

/*!
 * \brief Typed live view of an Op attribute column.
 *
 * Copies share the retained mutable column and observe later registrations,
 * replacements, resets, and growth. Values are returned by value, never as
 * references into a backing array. Registration is not synchronized with readers.
 * \tparam ValueType The attribute value type.
 */
template <typename ValueType>
class OpAttrMap {
 public:
  /*!
   * \brief Check whether an operator has a value in this column.
   * \param op The operator.
   * \return 1 if present, 0 otherwise.
   */
  int count(const Op& op) const {
    return op->index_ < column_.size() && column_[op->index_] != nullptr;
  }
  /*!
   * \brief Look up an operator's attribute, raising InternalError if absent.
   * \param op The operator.
   * \return The value converted to ValueType.
   */
  ValueType operator[](const Op& op) const {
    TVM_FFI_ICHECK(count(op)) << "Attribute " << attr_name_ << " has not been registered for "
                              << op->name;
    if constexpr (std::is_same_v<ValueType, ffi::Any>) {
      return column_[op->index_];
    } else {
      return column_[op->index_].template cast<ValueType>();
    }
  }
  /*!
   * \brief Look up an operator's attribute with a fallback.
   * \param op The operator.
   * \param def_value Fallback when the attribute is absent.
   * \return The registered value or def_value.
   */
  ValueType get(const Op& op, ValueType def_value) const {
    return count(op) ? (*this)[op] : def_value;
  }
  /*!
   * \brief Look up an expression's attribute with a fallback.
   * \param expr A defined expression.
   * \param def_value Fallback when expr is not an Op or its attribute is absent.
   * \return The registered value or def_value.
   */
  ValueType get(const Expr& expr, ValueType def_value) const {
    TVM_FFI_ICHECK(expr.defined());
    if (const auto* op = expr.as<OpNode>()) {
      return get(ffi::GetRef<Op>(op), def_value);
    }
    return def_value;
  }

 private:
  friend class Op;
  OpAttrMap(ffi::List<ffi::Any> column, ffi::String attr_name)
      : column_(std::move(column)), attr_name_(std::move(attr_name)) {}
  ffi::List<ffi::Any> column_;
  ffi::String attr_name_;
};

template <typename ValueType>
OpAttrMap<ValueType> Op::GetAttrMap(const ffi::String& attr_name) {
  return OpAttrMap<ValueType>(GetAttrColumn(attr_name), attr_name);
}

namespace op_attr {
inline constexpr const char* kInferType = "FInferType";
inline constexpr const char* kFixedReturnType = "TFixedReturnType";
}  // namespace op_attr

}  // namespace tvm
#endif  // TVM_IR_OP_H_
