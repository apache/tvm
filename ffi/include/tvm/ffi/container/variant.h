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
 * \file tvm/ffi/container/variant.h
 * \brief Runtime variant container types.
 */
#ifndef TVM_FFI_CONTAINER_VARIANT_H_
#define TVM_FFI_CONTAINER_VARIANT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/container_details.h>
#include <tvm/ffi/optional.h>

#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief A typed variant container.
 *
 * A Variant is backed by Any container, with strong checks during construction.
 */
template <typename... V>
class Variant {
 public:
  static_assert(details::all_storage_enabled_v<V...>,
                "All types used in Variant<...> must be compatible with Any");
  /*
   * \brief Helper utility to check if the type can be contained in the variant
   */
  template <typename T>
  static constexpr bool variant_contains_v = (details::type_contains_v<V, T> || ...);
  /* \brief Helper utility for SFINAE if the type is part of the variant */
  template <typename T>
  using enable_if_variant_contains_t = std::enable_if_t<variant_contains_v<T>>;

  Variant(const Variant<V...>& other) : data_(other.data_) {}
  Variant(Variant<V...>&& other) : data_(std::move(other.data_)) {}

  TVM_FFI_INLINE Variant& operator=(const Variant<V...>& other) {
    data_ = other.data_;
    return *this;
  }

  TVM_FFI_INLINE Variant& operator=(Variant<V...>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  Variant(T other) : data_(std::move(other)) {}  // NOLINT(*)

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE Variant& operator=(T other) {
    data_ = std::move(other);
    return *this;
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE std::optional<T> as() const {
    return data_.as<T>();
  }

  /*
   * \brief Shortcut of as Object to cast to a const pointer when T is an Object.
   *
   * \tparam T The object type.
   * \return The requested pointer, returns nullptr if type mismatches.
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
  TVM_FFI_INLINE const T* as() const {
    return data_.as<const T*>().value_or(nullptr);
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE T get() const& {
    return data_.template cast<T>();
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE T get() && {
    return std::move(data_).template cast<T>();
  }

  TVM_FFI_INLINE std::string GetTypeKey() const { return data_.GetTypeKey(); }

 private:
  friend struct TypeTraits<Variant<V...>>;
  friend struct ObjectPtrHash;
  friend struct ObjectPtrEqual;
  // constructor from any
  explicit Variant(Any data) : data_(std::move(data)) {}
  // internal data is backed by Any
  Any data_;
  /*!
   * \brief Get the object pointer from the variant
   * \note This function is only available if all types used in Variant<...> are derived from
   * ObjectRef
   */
  TVM_FFI_INLINE Object* GetObjectPtrForHashEqual() const {
    constexpr bool all_object_v = (std::is_base_of_v<ObjectRef, V> && ...);
    static_assert(all_object_v,
                  "All types used in Variant<...> must be derived from ObjectRef "
                  "to enable ObjectPtrHash/ObjectPtrEqual");
    return details::AnyUnsafe::ObjectPtrFromAnyAfterCheck(data_);
  }
};

template <typename... V>
inline constexpr bool use_default_type_traits_v<Variant<V...>> = false;

template <typename... V>
struct TypeTraits<Variant<V...>> : public TypeTraitsBase {
  static TVM_FFI_INLINE void CopyToAnyView(const Variant<V...>& src, TVMFFIAny* result) {
    *result = AnyView(src.data_).CopyToTVMFFIAny();
  }

  static TVM_FFI_INLINE void MoveToAny(Variant<V...> src, TVMFFIAny* result) {
    *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src.data_));
  }

  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraitsBase::GetMismatchTypeInfo(src);
  }

  static TVM_FFI_INLINE bool CheckAnyStorage(const TVMFFIAny* src) {
    return (TypeTraits<V>::CheckAnyStorage(src) || ...);
  }

  static TVM_FFI_INLINE Variant<V...> CopyFromAnyStorageAfterCheck(const TVMFFIAny* src) {
    return Variant<V...>(Any(AnyView::CopyFromTVMFFIAny(*src)));
  }

  static TVM_FFI_INLINE Variant<V...> MoveFromAnyStorageAfterCheck(TVMFFIAny* src) {
    return Variant<V...>(details::AnyUnsafe::MoveTVMFFIAnyToAny(std::move(*src)));
  }

  static TVM_FFI_INLINE std::optional<Variant<V...>> TryConvertFromAnyView(const TVMFFIAny* src) {
    // fast path, storage is already in the right type
    if (CheckAnyStorage(src)) {
      return CopyFromAnyStorageAfterCheck(src);
    }
    // More expensive path, try to convert to each type, in order of declaration
    return TryVariantTypes<V...>(src);
  }

  template <typename VariantType, typename... Rest>
  static TVM_FFI_INLINE std::optional<Variant<V...>> TryVariantTypes(const TVMFFIAny* src) {
    if (auto opt_convert = TypeTraits<VariantType>::TryConvertFromAnyView(src)) {
      return Variant<V...>(*std::move(opt_convert));
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryVariantTypes<Rest...>(src);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return details::ContainerTypeStr<V...>("Variant"); }
};

template <typename... V>
TVM_FFI_INLINE size_t ObjectPtrHash::operator()(const Variant<V...>& a) const {
  return std::hash<Object*>()(a.GetObjectPtrForHashEqual());
}

template <typename... V>
TVM_FFI_INLINE bool ObjectPtrEqual::operator()(const Variant<V...>& a,
                                               const Variant<V...>& b) const {
  return a.GetObjectPtrForHashEqual() == b.GetObjectPtrForHashEqual();
}

namespace details {
template <typename... V, typename T>
inline constexpr bool type_contains_v<Variant<V...>, T> = (type_contains_v<V, T> || ...);
}  // namespace details
}  // namespace ffi

// Expose to the tvm namespace
// Rationale: convinience and no ambiguity
using ffi::Variant;
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_VARIANT_H_
