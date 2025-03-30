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
 * \file tvm/ffi/optional.h
 * \brief Runtime Optional container types.
 */
#ifndef TVM_FFI_OPTIONAL_H_
#define TVM_FFI_OPTIONAL_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>

#include <optional>
#include <string>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Optional data type in FFI.
 * \tparam T The underlying type of the optional.
 *
 * \note Compared to std::optional, Optional<ObjectRef>
 *       takes less storage as it used nullptr to represent nullopt.
 */
template <typename T, typename = void>
class Optional;

template <typename T>
inline constexpr bool is_optional_type_v = false;

template <typename T>
inline constexpr bool is_optional_type_v<Optional<T>> = true;

template <typename T>
inline constexpr bool use_ptr_based_optional_v = (
  std::is_base_of_v<ObjectRef, T> && !is_optional_type_v<T>
);

// Specialization for non-ObjectRef types.
// simply fallback to std::optional
template <typename T>
class Optional<T, std::enable_if_t<!use_ptr_based_optional_v<T>>> {
 public:
  // default constructors.
  Optional() = default;
  Optional(const Optional<T>& other) : data_(other.data_) {}
  Optional(Optional<T>&& other) : data_(std::move(other.data_)) {}
  Optional(std::optional<T> other) : data_(std::move(other)) {}
  Optional(std::nullopt_t) {}
  Optional<T>& operator=(const Optional<T>& other) {
    data_ = other.data_;
    return *this;
  }
  Optional<T>& operator=(Optional<T>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  // normal value handling.
  Optional(T other)  // NOLINT(*)
      : data_(std::move(other)) {}
  Optional<T>& operator=(T other) {
    data_ = std::move(other);
    return *this;
  }
  Optional<T>& operator=(std::nullopt_t) {
    data_ = std::nullopt;
    return *this;
  }

  const T& value() const & {
    if (!data_.has_value()) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return *data_;
  }

  T&& value() && {
    if (!data_.has_value()) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return std::move(*data_);
  }

  T value_or(T default_value) const { return data_.value_or(default_value); }

  explicit operator bool() const { return data_.has_value(); }

  bool has_value() const { return data_.has_value(); }

  bool operator==(const Optional<T>& other) const { return data_ == other.data_; }

  bool operator!=(const Optional<T>& other) const { return data_ != other.data_; }

  template <typename U>
  bool operator==(const U& other) const {
    return data_ == other;
  }

  template <typename U>
  bool operator!=(const U& other) const {
    return data_ != other;
  }

 private:
  std::optional<T> data_;

  template <typename, typename>
  friend struct TypeTraits;
  // keep unsafe dereference private
  const T operator*() const & noexcept {
    // it is OK to reinterpret_cast ObjectPtr<Object>& to ObjectRef&.
    return *data_;
  }
  T operator*() && noexcept {
     return T(std::move(*data_));
  }
};

// Specialization for ObjectRef types.
// nullptr is treated as std::nullopt.
template <typename T>
class Optional<T, std::enable_if_t<use_ptr_based_optional_v<T>>> : public ObjectRef {
 public:
  using ContainerType = typename T::ContainerType;
  Optional() = default;
  Optional(const Optional<T>& other) : ObjectRef(other.data_) {}
  Optional(Optional<T>&& other) : ObjectRef(std::move(other.data_)) {}
  Optional<T>& operator=(const Optional<T>& other) {
    data_ = other.data_;
    return *this;
  }
  Optional<T>& operator=(Optional<T>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  // nullopt hanlding
  Optional(std::nullopt_t) {}  // NOLINT(*)

  // handle conversion from std::optional<T>
  Optional(std::optional<T> other) {
    if (other.has_value()) {
      *this = std::move(*other);
    }
  }
  // normal value handling.
  Optional(T other)  // NOLINT(*)
      : ObjectRef(std::move(other)) {}

  Optional<T>& operator=(T other) {
    ObjectRef::operator=(std::move(other));
    return *this;
  }

  const T& value() const & {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return operator*();
  }

  T&& value() && {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return std::move(operator*());
  }

  T value_or(T default_value) const { return data_ != nullptr ? T(data_) : default_value; }

  explicit operator bool() const { return data_ != nullptr; }

  bool has_value() const { return data_ != nullptr; }

  // explicit disable nullptr comparison
  // so we can canonically always access optional via has_value() and value()
  bool operator==(std::nullptr_t) = delete;
  bool operator!=(std::nullptr_t) = delete;

  // operator overloadings
  auto operator==(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    return EQToOptional(other);
  }
  auto operator!=(const Optional<T>& other) const {
    return NEToOptional(other);
  }

  auto operator==(const std::optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    return EQToOptional(other);
  }
  auto operator!=(const std::optional<T>& other) const {
    return NEToOptional(other);
  }

  auto operator==(const T& other) const {
    using RetType = decltype(value() == other);
    if (same_as(other)) return RetType(true);
    if (has_value()) return operator*() == other;
    return RetType(false);
  }

  auto operator!=(const T& other) const { return !(*this == other); }

  template <typename U>
  auto operator==(const U& other) const {
    using RetType = decltype(value() == other);
    if (!has_value()) return RetType(false);
    return operator*() == other;
  }

  template <typename U>
  auto operator!=(const U& other) const {
    using RetType = decltype(value() != other);
    if (!has_value()) return RetType(true);
    return operator*() != other;
  }

  static constexpr bool _type_is_nullable = true;

 private:
  template <typename, typename>
  friend struct TypeTraits;
  // hide the object ptr constructor and make it only accessible
  // to friend TypeTraits
  explicit Optional(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  // inherit get method as private
  using ObjectRef::get;
  using ObjectRef::defined;
  // keep unsafe dereference private
  TVM_FFI_INLINE const T& operator*() const & noexcept {
    // safe to reinterpret_cast ObjectPtr<Object>& to ObjectRef&.
    return reinterpret_cast<const T&>(data_);
  }
  TVM_FFI_INLINE T operator*() && noexcept {
     return T(std::move(data_));
  }
  template<typename U>
  TVM_FFI_INLINE auto EQToOptional(const U& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() == other.value());
    if (same_as(other)) return RetType(true);
    if (has_value() && other.has_value()) {
      return operator*() == *other;
    } else {
      // one of them is nullptr.
      return RetType(false);
    }
  }

  template<typename U>
  TVM_FFI_INLINE auto NEToOptional(const U& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() != other.value());
    if (same_as(other)) return RetType(false);
    if (has_value() && other.has_value()) {
      return operator*() != *other;
    } else {
      // one of them is nullptr.
      return RetType(true);
    }
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_OPTIONAL_H_
