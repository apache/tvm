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
 * \file tvm/ffi/container/optional.h
 * \brief Runtime Optional container types.
 */
#ifndef TVM_FFI_CONTAINER_OPTIONAL_H_
#define TVM_FFI_CONTAINER_OPTIONAL_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/object.h>

#include <optional>
#include <string>
#include <utility>

namespace tvm {
namespace ffi {

template <typename T, typename = void>
class Optional;

/*!
 * \brief Optional that is backed by Any
 *
 * nullptr will be treated as NullOpt
 *
 * \tparam T any value will be treated as
 */
template <typename T>
class Optional<T, std::enable_if_t<!std::is_base_of_v<ObjectRef, T>>> {
 public:
  static_assert(!std::is_same_v<std::nullptr_t, T>, "Optional<nullptr> is not well defined");
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
  // nullptr handling.
  // disallow implicit conversion as 0 can be implicitly converted to nullptr_t
  explicit Optional(std::nullptr_t) {}
  Optional<T>& operator=(std::nullptr_t) {
    data_ = std::nullopt;
    return *this;
  }
  Optional<T>& operator=(std::nullopt_t) {
    data_ = std::nullopt;
    return *this;
  }
  /*!
   * \return A not-null container value in the optional.
   * \note This function performs not-null checking.
   */
  T value() const {
    if (!data_.has_value()) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return *data_;
  }
  /*!
   * \return A not-null container value in the optional.
   * \note This function performs not-null checking.
   */
  T value_or(T default_value) const { return data_.value_or(default_value); }

  /*! \return Whether the container is not nullptr.*/
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

  // operator overloadings with nullptr
  bool operator==(std::nullptr_t) const { return !data_.has_value(); }
  bool operator!=(std::nullptr_t) const { return data_.has_value(); }

  // helper function to move out value
  T MoveValueNoCheck() { return std::move(data_.value()); }
  // helper function to copy out value
  T CopyValueNoCheck() const { return *data_; }

 private:
  std::optional<T> data_;
};

/*!
 * \brief Specialization of Optional for ObjectRef.
 *
 * In such cases, nullptr is treated as NullOpt.
 * This specialization reduces the storage cost of
 * Optional for ObjectRef.
 *
 * \tparam T The original ObjectRef.
 */
template <typename T>
class Optional<T, std::enable_if_t<std::is_base_of_v<ObjectRef, T>>> : public ObjectRef {
 public:
  using ContainerType = typename T::ContainerType;
  static_assert(std::is_base_of<ObjectRef, T>::value, "Optional is only defined for ObjectRef.");
  // default constructors.
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
  /*!
   * \brief Construct from an ObjectPtr
   *        whose type already matches the ContainerType.
   * \param ptr
   */
  explicit Optional(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  /*! \brief Nullopt handling */
  Optional(std::nullopt_t) {}  // NOLINT(*)
  // nullptr handling.
  // disallow implicit conversion as 0 can be implicitly converted to nullptr_t
  explicit Optional(std::nullptr_t) {}
  Optional<T>& operator=(std::nullptr_t) {
    data_ = nullptr;
    return *this;
  }
  // handle conversion from std::optional<T>
  Optional(std::optional<T> other) {
    if (other.has_value()) {
      *this = std::move(other.value());
    }
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
  // helper function to move out value
  T&& MoveOutValueNoCheck() { return T(std::move(data_)); }
  /*!
   * \return A not-null container value in the optional.
   * \note This function performs not-null checking.
   */
  T value() const {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Bad optional access";
    }
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
  /*! \return Whether the container is not nullptr */
  bool has_value() const { return *this != nullptr; }

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

  // helper function to move out value
  T MoveValueNoCheck() { return T(std::move(data_)); }
  // helper function to copy out value
  T CopyValueNoCheck() const { return T(data_); }
};

template <typename T>
inline constexpr bool use_default_type_traits_v<Optional<T>> = false;

template <typename T>
struct TypeTraits<Optional<T>> : public TypeTraitsBase {
  static TVM_FFI_INLINE void CopyToAnyView(const Optional<T>& src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::CopyToAnyView(src.CopyValueNoCheck(), result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }

  static TVM_FFI_INLINE void MoveToAny(Optional<T> src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::MoveToAny(src.MoveValueNoCheck(), result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }

  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraits<T>::GetMismatchTypeInfo(src);
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return true;
    return TypeTraits<T>::CheckAnyView(src);
  }

  static TVM_FFI_INLINE Optional<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Optional<T>(nullptr);
    return TypeTraits<T>::CopyFromAnyViewAfterCheck(src);
  }

  static TVM_FFI_INLINE std::optional<Optional<T>> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Optional<T>(nullptr);
    return TypeTraits<T>::TryCopyFromAnyView(src);
  }

  static TVM_FFI_INLINE std::string TypeStr() {
    return "Optional<" + TypeTraits<T>::TypeStr() + ">";
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_OPTIONAL_H_
