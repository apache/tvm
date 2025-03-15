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
 * \file tvm/runtime/container/optional.h
 * \brief Runtime Optional container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_OPTIONAL_H_
#define TVM_RUNTIME_CONTAINER_OPTIONAL_H_

#include <utility>

#include "./base.h"

namespace tvm {
namespace runtime {

/*! \brief Helper to represent nullptr for optional. */
struct NullOptType {};

/*!
 * \brief Optional container that to represent to a Nullable variant of T.
 * \tparam T The original ObjectRef.
 *
 * \code
 *
 *  Optional<String> opt0 = nullptr;
 *  Optional<String> opt1 = String("xyz");
 *  ICHECK(opt0 == nullptr);
 *  ICHECK(opt1 == "xyz");
 *
 * \endcode
 */
template <typename T>
class Optional : public ObjectRef {
 public:
  using ContainerType = typename T::ContainerType;
  static_assert(std::is_base_of<ObjectRef, T>::value, "Optional is only defined for ObjectRef.");
  // default constructors.
  Optional() = default;
  Optional(const Optional<T>&) = default;
  Optional(Optional<T>&&) = default;
  Optional<T>& operator=(const Optional<T>&) = default;
  Optional<T>& operator=(Optional<T>&&) = default;
  /*!
   * \brief Construct from an ObjectPtr
   *        whose type already matches the ContainerType.
   * \param ptr
   */
  explicit Optional(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  /*! \brief Nullopt handling */
  Optional(NullOptType) {}  // NOLINT(*)
  // nullptr handling.
  // disallow implicit conversion as 0 can be implicitly converted to nullptr_t
  explicit Optional(std::nullptr_t) {}
  Optional<T>& operator=(std::nullptr_t) {
    data_ = nullptr;
    return *this;
  }
  // normal value handling.
  Optional(T other)  // NOLINT(*)
      : ObjectRef(std::move(other)) {}
  Optional<T>& operator=(T other) {
    ObjectRef::operator=(std::move(other));
    return *this;
  }
  // delete the int constructor
  // since Optional<Integer>(0) is ambiguious
  // 0 can be implicitly casted to nullptr_t
  explicit Optional(int val) = delete;
  Optional<T>& operator=(int val) = delete;
  /*!
   * \return A not-null container value in the optional.
   * \note This function performs not-null checking.
   */
  T value() const {
    ICHECK(data_ != nullptr);
    return T(data_);
  }
  /*!
   * \return The internal object pointer with container type of T.
   * \note This function do not perform not-null checking.
   */
  const ContainerType* get() const { return static_cast<ContainerType*>(data_.get()); }
  /*!
   * \return The contained value if the Optional is not null
   *         otherwise return the default_value.
   */
  T value_or(T default_value) const { return data_ != nullptr ? T(data_) : default_value; }

  /*! \return Whether the container is not nullptr.*/
  explicit operator bool() const { return *this != nullptr; }
  // operator overloadings
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }
  auto operator==(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() == other.value());
    if (same_as(other)) return RetType(true);
    if (*this != nullptr && other != nullptr) {
      return value() == other.value();
    } else {
      // one of them is nullptr.
      return RetType(false);
    }
  }
  auto operator!=(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() != other.value());
    if (same_as(other)) return RetType(false);
    if (*this != nullptr && other != nullptr) {
      return value() != other.value();
    } else {
      // one of them is nullptr.
      return RetType(true);
    }
  }
  auto operator==(const T& other) const {
    using RetType = decltype(value() == other);
    if (same_as(other)) return RetType(true);
    if (*this != nullptr) return value() == other;
    return RetType(false);
  }
  auto operator!=(const T& other) const { return !(*this == other); }
  template <typename U>
  auto operator==(const U& other) const {
    using RetType = decltype(value() == other);
    if (*this == nullptr) return RetType(false);
    return value() == other;
  }
  template <typename U>
  auto operator!=(const U& other) const {
    using RetType = decltype(value() != other);
    if (*this == nullptr) return RetType(true);
    return value() != other;
  }
  static constexpr bool _type_is_nullable = true;
};

template <typename ObjectRefType, typename>
inline Optional<ObjectRefType> ObjectRef::as() const {
  if (auto* ptr = this->as<typename ObjectRefType::ContainerType>()) {
    return GetRef<ObjectRefType>(ptr);
  } else {
    return NullOptType{};
  }
}

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::Optional;
constexpr runtime::NullOptType NullOpt{};
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_OPTIONAL_H_
