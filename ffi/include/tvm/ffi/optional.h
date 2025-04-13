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

template <typename T>
inline constexpr bool is_optional_type_v = false;

template <typename T>
inline constexpr bool is_optional_type_v<Optional<T>> = true;

// we can safely used ptr based optional for ObjectRef types
// that do not have additional data members and virtual functions.
template <typename T>
inline constexpr bool use_ptr_based_optional_v =
    (std::is_base_of_v<ObjectRef, T> && !is_optional_type_v<T>);

// Specialization for non-ObjectRef types.
// simply fallback to std::optional
template <typename T>
class Optional<T, std::enable_if_t<!use_ptr_based_optional_v<T>>> {
 public:
  // default constructors.
  Optional() = default;
  Optional(const Optional<T>& other) : data_(other.data_) {}
  Optional(Optional<T>&& other) : data_(std::move(other.data_)) {}
  Optional(std::optional<T> other) : data_(std::move(other)) {}  // NOLINT(*)
  Optional(std::nullopt_t) {}                                    // NOLINT(*)
  // normal value handling.
  Optional(T other)  // NOLINT(*)
      : data_(std::move(other)) {}

  TVM_FFI_INLINE Optional<T>& operator=(const Optional<T>& other) {
    data_ = other.data_;
    return *this;
  }

  TVM_FFI_INLINE Optional<T>& operator=(Optional<T>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  TVM_FFI_INLINE Optional<T>& operator=(T other) {
    data_ = std::move(other);
    return *this;
  }

  TVM_FFI_INLINE Optional<T>& operator=(std::nullopt_t) {
    data_ = std::nullopt;
    return *this;
  }

  TVM_FFI_INLINE const T& value() const& {
    if (!data_.has_value()) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return *data_;
  }

  TVM_FFI_INLINE T&& value() && {
    if (!data_.has_value()) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return *std::move(data_);
  }

  template <typename U = std::remove_cv_t<T>>
  TVM_FFI_INLINE T value_or(U&& default_value) const {
    return data_.value_or(std::forward<U>(default_value));
  }

  TVM_FFI_INLINE explicit operator bool() const noexcept { return data_.has_value(); }

  TVM_FFI_INLINE bool has_value() const noexcept { return data_.has_value(); }

  TVM_FFI_INLINE bool operator==(const Optional<T>& other) const { return data_ == other.data_; }

  TVM_FFI_INLINE bool operator!=(const Optional<T>& other) const { return data_ != other.data_; }

  template <typename U>
  TVM_FFI_INLINE bool operator==(const U& other) const {
    return data_ == other;
  }
  template <typename U>
  TVM_FFI_INLINE bool operator!=(const U& other) const {
    return data_ != other;
  }

  /*!
   * \brief Direct access to the value.
   * \return the xvalue reference to the stored value.
   * \note only use this function after checking has_value()
   */
  TVM_FFI_INLINE T&& operator*() && noexcept { return *std::move(data_); }
  /*!
   * \brief Direct access to the value.
   * \return the const reference to the stored value.
   * \note only use this function  after checking has_value()
   */
  TVM_FFI_INLINE const T& operator*() const& noexcept { return *data_; }

 private:
  std::optional<T> data_;
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
  explicit Optional(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  // nullopt hanlding
  Optional(std::nullopt_t) {}  // NOLINT(*)

  // handle conversion from std::optional<T>
  Optional(std::optional<T> other) {  // NOLINT(*)
    if (other.has_value()) {
      *this = *std::move(other);
    }
  }
  // normal value handling.
  Optional(T other)  // NOLINT(*)
      : ObjectRef(std::move(other)) {}

  TVM_FFI_INLINE Optional<T>& operator=(T other) {
    ObjectRef::operator=(std::move(other));
    return *this;
  }

  TVM_FFI_INLINE Optional<T>& operator=(const Optional<T>& other) {
    data_ = other.data_;
    return *this;
  }

  TVM_FFI_INLINE Optional<T>& operator=(std::nullptr_t) {
    data_ = nullptr;
    return *this;
  }

  TVM_FFI_INLINE Optional<T>& operator=(Optional<T>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  TVM_FFI_INLINE T value() const& {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return T(data_);
  }

  TVM_FFI_INLINE T value() && {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return T(std::move(data_));
  }

  template <typename U = std::remove_cv_t<T>>
  TVM_FFI_INLINE T value_or(U&& default_value) const {
    return data_ != nullptr ? T(data_) : T(std::forward<U>(default_value));
  }

  TVM_FFI_INLINE explicit operator bool() const { return data_ != nullptr; }

  TVM_FFI_INLINE bool has_value() const { return data_ != nullptr; }

  /*!
   * \brief Direct access to the value.
   * \return the const reference to the stored value.
   * \note only use this function after checking has_value()
   */
  TVM_FFI_INLINE T operator*() const& noexcept { return T(data_); }

  /*!
   * \brief Direct access to the value.
   * \return the const reference to the stored value.
   * \note only use this function  after checking has_value()
   */
  TVM_FFI_INLINE T operator*() && noexcept { return T(std::move(data_)); }

  TVM_FFI_INLINE bool operator==(std::nullptr_t) const noexcept { return !has_value(); }
  TVM_FFI_INLINE bool operator!=(std::nullptr_t) const noexcept { return has_value(); }

  // operator overloadings
  TVM_FFI_INLINE auto operator==(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    return EQToOptional(other);
  }
  TVM_FFI_INLINE auto operator!=(const Optional<T>& other) const { return NEToOptional(other); }

  TVM_FFI_INLINE auto operator==(const std::optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    return EQToOptional(other);
  }
  TVM_FFI_INLINE auto operator!=(const std::optional<T>& other) const {
    return NEToOptional(other);
  }

  TVM_FFI_INLINE auto operator==(const T& other) const {
    using RetType = decltype(value() == other);
    if (same_as(other)) return RetType(true);
    if (has_value()) return operator*() == other;
    return RetType(false);
  }

  TVM_FFI_INLINE auto operator!=(const T& other) const { return !(*this == other); }

  template <typename U>
  TVM_FFI_INLINE auto operator==(const U& other) const {
    using RetType = decltype(value() == other);
    if (!has_value()) return RetType(false);
    return operator*() == other;
  }

  template <typename U>
  TVM_FFI_INLINE auto operator!=(const U& other) const {
    using RetType = decltype(value() != other);
    if (!has_value()) return RetType(true);
    return operator*() != other;
  }

  /*!
   * \return The internal object pointer with container type of T.
   * \note This function do not perform not-null checking.
   */
  TVM_FFI_INLINE const ContainerType* get() const {
    return static_cast<ContainerType*>(data_.get());
  }

 private:
  template <typename U>
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

  template <typename U>
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
