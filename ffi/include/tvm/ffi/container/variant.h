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

#include <sstream>
#include <type_traits>

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
  static constexpr bool all_compatible_with_any_v = (TypeTraits<V>::enabled && ...);
  static_assert(all_compatible_with_any_v,
                "All types used in Variant<...> must be compatible with Any");

  /*
   * \brief Helper utility to check if the type is part of the variant
   * \note Need is_same_v for POD types here.
   */
  template <typename T>
  static constexpr bool is_variant_v =
      ((std::is_base_of_v<V, T> || ...) || (std::is_same_v<T, V> || ...));
  /* \brief Helper utility for SFINAE if the type is part of the variant */
  template <typename T>
  using enable_if_variant_t = std::enable_if_t<is_variant_v<T>>;

  template <typename T, typename = enable_if_variant_t<T>>
  Variant(T other) : data_(std::move(other)) {}

  template <typename T, typename = enable_if_variant_t<T>>
  TVM_FFI_INLINE Variant& operator=(T other) {
    data_ = std::move(other);
    return *this;
  }

  template <typename T, typename = enable_if_variant_t<T>>
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

  template <typename T, typename = enable_if_variant_t<T>>
  TVM_FFI_INLINE T Get() const {
    return data_.operator T();
  }

 private:
  friend struct TypeTraits<Variant<V...>>;
  friend struct ObjectPtrHash;
  friend struct ObjectPtrEqual;
  // constructor from any
  Variant(Any data) : data_(std::move(data)) {}
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
    return details::AnyUnsafe::GetObjectPtrFromAny(data_);
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
    details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src.data_), result);
  }

  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraitsBase::GetMismatchTypeInfo(src);
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return (TypeTraits<V>::CheckAnyView(src) || ...);
  }

  static TVM_FFI_INLINE Variant<V...> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return Variant<V...>(Any(AnyView::CopyFromTVMFFIAny(*src)));
  }

  static TVM_FFI_INLINE std::optional<Variant<V...>> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyView(src)) {
      return CopyFromAnyViewAfterCheck(src);
    } else {
      return std::nullopt;
    }
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

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_VARIANT_H_
