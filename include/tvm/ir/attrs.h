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
 * \file tvm/ir/attrs.h
 * \brief Helpers for attribute objects.
 *
 *  This module enables declaration of named attributes
 *  which support default value setup and bound checking.
 *
 * \sa AttrsNode
 */
#ifndef TVM_IR_ATTRS_H_
#define TVM_IR_ATTRS_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cow.h>
#include <tvm/ir/expr.h>

#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace tvm {

/*!
 * \brief Base class of all attribute class
 * \sa Attrs
 */
class AttrsNode : public ffi::Object {
 public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("ir.Attrs", AttrsNode, ffi::Object);
};

/*!
 * \brief Managed reference to AttrsNode.
 * \sa AttrsNode
 */
class Attrs : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Attrs, ffi::ObjectRef, AttrsNode);
};

/*!
 * \brief Specialized attribute type that is backed by a map.
 *  The DictAttrsNode implements the Attrs behavior,
 *  its fields are directly accessible via object.field_name
 *  like other normal nodes.
 */
class DictAttrsNode : public AttrsNode {
 public:
  /*! \brief internal attrs map */
  ffi::Map<ffi::String, ffi::Any> dict;

  static void RegisterReflection() {
    namespace rfl = ffi::reflection;
    rfl::ObjectDef<DictAttrsNode>().def_ro("__dict__", &DictAttrsNode::dict);
  }

  // type info
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.DictAttrs", DictAttrsNode, AttrsNode);
};

/*!
 * \brief Managed reference to DictAttrsNode
 * \sa DictAttrsNode.
 *
 * \note DictAttrs is NOTNULLABLE: every instance must hold a backing
 *       DictAttrsNode. The class enforces this end-to-end by:
 *       - the default constructor (no args) allocating an empty backing,
 *       - the copy/move ctors and assignments leaving the moved-from
 *         instance in a defined-but-empty state rather than null,
 *       - the FFI type traits rejecting None at deserialization boundaries
 *         (since `_type_is_nullable == false`), and
 *       - the FFI lambda for ``ir.IRModule`` explicitly normalizing a
 *         missing/None attrs argument to ``DictAttrs()`` before forwarding
 *         to the C++ constructor.
 *       Callers (including third-party code via templates like ``WithAttr``)
 *       can therefore rely on ``attrs->dict`` being safe to dereference
 *       without a ``.defined()`` guard.
 */
class DictAttrs : public Attrs {
 public:
  /*!
   * \brief Construct a DictAttrs backed by DictAttrsNode.
   *
   * The no-argument form constructs an empty (but always defined) DictAttrs.
   * \param dict The attributes.
   */
  explicit DictAttrs(ffi::Map<ffi::String, Any> dict = {}) {
    ffi::ObjectPtr<DictAttrsNode> n = ffi::make_object<DictAttrsNode>();
    n->dict = std::move(dict);
    data_ = std::move(n);
  }

  /*!
   * \brief Move constructor that leaves the source in a defined-but-empty
   *        state rather than null, preserving the NOTNULLABLE invariant
   *        even after `std::move`.
   */
  DictAttrs(DictAttrs&& other) noexcept : Attrs(ffi::UnsafeInit{}) {
    data_ = std::move(other.data_);
    other.data_ = ffi::make_object<DictAttrsNode>();
  }

  /*!
   * \brief Move assignment that leaves the source in a defined-but-empty
   *        state rather than null, preserving the NOTNULLABLE invariant
   *        even after `std::move`.
   */
  DictAttrs& operator=(DictAttrs&& other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      other.data_ = ffi::make_object<DictAttrsNode>();
    }
    return *this;
  }

  // Explicit copy ctor/assign defaults. Declaring the move members above
  // would otherwise suppress the implicit copy members.
  DictAttrs(const DictAttrs& other) = default;
  DictAttrs& operator=(const DictAttrs& other) = default;

  // Utils for accessing attributes
  /*!
   * \brief Get a function attribute.
   *
   * \param attr_key The attribute key.
   * \param default_value The default value if the key does not exist, defaults to nullptr.
   *
   * \return The result
   *
   * \tparam TOBjectRef the expected object type.
   * \throw Error if the key exists but the value does not match TObjectRef
   *
   * \code
   *
   *  void GetAttrExample(const BaseFunc& f) {
   *    auto value = f->attrs.GetAttr<int64_t>("AttrKey", 0);
   *  }
   *
   * \endcode
   */
  template <typename TObjectRef>
  ffi::Optional<TObjectRef> GetAttr(
      const std::string& attr_key,
      ffi::Optional<TObjectRef> default_value = ffi::Optional<TObjectRef>(std::nullopt)) const {
    const DictAttrsNode* node = get();
    auto it = node->dict.find(attr_key);
    if (it != node->dict.end()) {
      return (*it).second.cast<TObjectRef>();
    } else {
      return default_value;
    }
  }
  // variant that uses TObjectRef to enable implicit conversion to default value.
  template <typename TObjectRef>
  ffi::Optional<TObjectRef> GetAttr(const std::string& attr_key, TObjectRef default_value) const {
    return GetAttr<TObjectRef>(attr_key, ffi::Optional<TObjectRef>(default_value));
  }
  /*!
   * \brief Check whether the function has an non-zero integer attr.
   *
   * This function can be used to check whether an optional
   * attribute mark(e.g. inline) exists.
   *
   * \param attr_key The key to the attribute.
   * \return The check result.
   *
   * \code
   *
   *  void HasNonzeroAttrExample(const BaseFunc& f) {
   *    if (f->HasNonzeroAttr(attr::kInline)) {
   *      // inline the function.
   *    }
   *  }
   *
   * \endcode
   */
  bool HasNonzeroAttr(const std::string& attr_key) const {
    return GetAttr<int64_t>(attr_key, 0).value_or(0) != 0;
  }

  // Inline-expand TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE here, minus
  // the default copy/move it normally injects (we define our own move members
  // above so the moved-from instance stays defined-but-empty).
  explicit DictAttrs(::tvm::ffi::UnsafeInit tag) : Attrs(tag) {}
  using __PtrType =
      std::conditional_t<DictAttrsNode::_type_mutable, DictAttrsNode*, const DictAttrsNode*>;
  __PtrType operator->() const { return static_cast<__PtrType>(data_.get()); }
  __PtrType get() const { return static_cast<__PtrType>(data_.get()); }
  static constexpr bool _type_is_nullable = false;
  using ContainerType = DictAttrsNode;
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DictAttrsNode);
};

/*!
 * \brief Copy the function or module, but overrides
 *        the attribute value key with the value.
 *
 * \param input The thing to annotate (BaseFunc or IRModule)
 * \param attr_key The attribute key.
 * \param attr_value The value attribute value.
 *
 * \tparam TFunc The corresponding function or module type.
 *
 * \returns The new function or module with updated attributes.
 *
 * \note This function performs copy on write optimization for func and module.
 *       If we move a uniquely referenced func or module into WithAttr,
 *       then no additional copy will be performed.
 *
 *       This is also why we make it as a function instead of a member function
 *       and why we pass by value in the first argument.
 *
 * \code
 *
 *  // Recommended way to trigger copy on write
 *  func = WithAttr(std::move(func), "key1", value1);
 *  func = WithAttr(std::move(func), "key2", value2);
 *
 * \endcode
 */
template <typename TFunc>
inline TFunc WithAttr(TFunc input, const std::string& attr_key, Any attr_value) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = input.CopyOnWrite();
  // node->attrs is NOTNULLABLE by contract, but defend against a caller
  // that left a moved-from DictAttrs in place by re-initializing here.
  if (!node->attrs.defined()) node->attrs = DictAttrs();
  node->attrs.CopyOnWrite()->dict.Set(attr_key, std::move(attr_value));
  return input;
}

/*!
 * \brief Copy the function or module, but overrides the attributes with the entries from \p attrs.
 *
 * \param input The thing to annotate (BaseFunc or IRModule)
 * \param attrs Key/values attributes to add to \p input.
 *
 * \tparam TFunc The corresponding function or module type.
 *
 * \returns The new function or module with updated attributes.
 */
template <typename TFunc>
inline TFunc WithAttrs(TFunc input, ffi::Map<ffi::String, Any> attrs) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  if (attrs.empty()) return input;
  TNode* node = input.CopyOnWrite();
  // node->attrs is NOTNULLABLE by contract, but defend against a caller
  // that left a moved-from DictAttrs in place by re-initializing here.
  if (!node->attrs.defined()) node->attrs = DictAttrs();
  auto* dict_node = node->attrs.CopyOnWrite();
  for (const auto& [k, v] : attrs) {
    dict_node->dict.Set(k, v);
  }
  return input;
}

/*!
 * \brief Copy the function or module, but removes the specified
 *        attribute.
 *
 * \param input The thing to annotate (BaseFunc or IRModule)
 * \param attr_key The attribute key.
 *
 * \tparam TFunc The corresponding function or module type.
 *
 * \returns The new function or module with removed attribute.
 *
 * \note This function performs copy on write optimization for func and module.
 *       If we move a uniquely referenced func or module into WithoutAttr,
 *       then no additional copy will be performed.
 *
 *       This is also why we make it as a function instead of a member function
 *       and why we pass by value in the first argument.
 *
 * \code
 *
 *  // Recommended way to trigger copy on write
 *  func = WithoutAttr(std::move(func), "key1");
 *  func = WithoutAttr(std::move(func), "key2");
 *
 * \endcode
 */
template <typename TFunc>
inline TFunc WithoutAttr(TFunc input, const std::string& attr_key) {
  using TNode = typename TFunc::ContainerType;
  static_assert(TNode::_type_final, "Can only operate on the leaf nodes");
  TNode* node = input.CopyOnWrite();
  // node->attrs is NOTNULLABLE by contract, but defend against a caller
  // that left a moved-from DictAttrs in place; nothing to erase from an
  // empty dict.
  if (!node->attrs.defined()) {
    node->attrs = DictAttrs();
    return input;
  }
  node->attrs.CopyOnWrite()->dict.erase(attr_key);
  return input;
}

}  // namespace tvm
#endif  // TVM_IR_ATTRS_H_
