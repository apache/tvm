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
 * \file tvm/runtime/container/variant.h
 * \brief Runtime Variant container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_VARIANT_H_
#define TVM_RUNTIME_CONTAINER_VARIANT_H_

#include <tvm/runtime/object.h>

#include <tuple>
#include <type_traits>
#include <utility>

namespace tvm {
namespace runtime {

namespace detail {
template <typename Parent, typename ChildTuple>
constexpr bool parent_is_base_of_any = false;

template <typename Parent, typename... Child>
constexpr bool parent_is_base_of_any<Parent, std::tuple<Child...>> =
    ((std::is_base_of_v<Parent, Child> && !std::is_same_v<Parent, Child>) || ...);

/* \brief Utility to check if any parent is a base class of any child
 *
 * The type-checking in Variant relies on all types being from
 * independent types, such that `Object::IsInstance` is sufficient to
 * determine which variant is populated.
 *
 * For example, suppose the illegal `Variant<tir::Var, tir::PrimExpr>`
 * were allowed (e.g. to represent either the defintion of a variable
 * or the usage of a variable).  If a function returned
 * `tir::PrimExpr`, it could result in either variant being filled, as
 * the underlying type at runtime could be a `tir::Var`.  This
 * behavior is different from `std::variant`, which determines the
 * active variant based solely on the compile-time type, and could
 * produce very unexpected results if the variants have different
 * semantic interpretations.
 */
template <typename ParentTuple, typename ChildTuple>
static constexpr bool any_parent_is_base_of_any_child = false;

template <typename ChildTuple, typename... Parent>
static constexpr bool any_parent_is_base_of_any_child<std::tuple<Parent...>, ChildTuple> =
    (parent_is_base_of_any<Parent, ChildTuple> || ...);
}  // namespace detail

template <typename... V>
class Variant : public ObjectRef {
  static constexpr bool all_inherit_from_objectref = (std::is_base_of_v<ObjectRef, V> && ...);
  static_assert(all_inherit_from_objectref,
                "All types used in Variant<...> must inherit from ObjectRef");

  static constexpr bool a_variant_inherits_from_another_variant =
      detail::any_parent_is_base_of_any_child<std::tuple<V...>, std::tuple<V...>>;
  static_assert(!a_variant_inherits_from_another_variant,
                "Due to implementation limitations, "
                "no type stored in a tvm::runtime::Variant "
                "may be a subclass of any other type "
                "stored in the same variant.");

 public:
  /* \brief Helper utility to check if the type is part of the variant */
  template <typename T>
  static constexpr bool is_variant = (std::is_base_of_v<V, T> || ...);

  /* \brief Helper utility for SFINAE if the type is part of the variant */
  template <typename T>
  using enable_if_variant = std::enable_if_t<is_variant<T>>;

  template <typename T, typename = enable_if_variant<T>>
  Variant(T value) : ObjectRef(std::move(value)) {}  // NOLINT(*)

  template <typename T, typename = enable_if_variant<T>>
  Variant& operator=(T value) {
    ObjectRef::operator=(std::move(value));
    return *this;
  }

  // These functions would normally be declared with the
  // TVM_DEFINE_OBJECT_REF_METHODS macro.  However, we need additional
  // type-checking inside the ObjectPtr<Object> constructor.
  using ContainerType = Object;
  Variant() : ObjectRef() {}
  explicit Variant(ObjectPtr<Object> node) : ObjectRef(node) {
    CHECK(node == nullptr || (node->IsInstance<typename V::ContainerType>() || ...))
        << "Variant<"
        << static_cast<const std::stringstream&>(
               (std::stringstream() << ... << V::ContainerType::_type_key))
               .str()
        << "> cannot hold an object of type " << node->GetTypeKey();
  }
  TVM_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(Variant);
};

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::Variant;

}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_VARIANT_H_
