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
namespace details {
/*!
 * \brief Base class for Variant.
 *
 * \tparam all_storage_object Whether all types are derived from ObjectRef.
 */
template <bool all_storage_object = false>
class VariantBase {
 public:
  TVM_FFI_INLINE bool same_as(const VariantBase<all_storage_object>& other) const {
    return data_.same_as(other.data_);
  }

 protected:
  template <typename T>
  explicit VariantBase(T other) : data_(std::move(other)) {}

  TVM_FFI_INLINE void SetData(Any other_data) { data_ = std::move(other_data); }

  TVM_FFI_INLINE Any MoveToAny() && { return std::move(data_); }

  TVM_FFI_INLINE AnyView ToAnyView() const { return data_.operator AnyView(); }

  Any data_;
};

// Specialization for all object ref case, backed by ObjectRef.
template <>
class VariantBase<true> : public ObjectRef {
 protected:
  template <typename T>
  explicit VariantBase(const T& other) : ObjectRef(other) {}
  template <typename T>
  explicit VariantBase(T&& other) : ObjectRef(std::move(other)) {}
  explicit VariantBase(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  explicit VariantBase(Any other)
      : ObjectRef(details::AnyUnsafe::MoveFromAnyAfterCheck<ObjectRef>(std::move(other))) {}

  TVM_FFI_INLINE void SetData(ObjectPtr<Object> other) { data_ = std::move(other); }

  TVM_FFI_INLINE Any MoveToAny() && { return Any(ObjectRef(std::move(data_))); }

  TVM_FFI_INLINE AnyView ToAnyView() const {
    TVMFFIAny any_data;
    if (data_ == nullptr) {
      any_data.type_index = TypeIndex::kTVMFFINone;
      any_data.v_int64 = 0;
    } else {
      TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&any_data);
      any_data.type_index = data_->type_index();
      any_data.v_obj = details::ObjectUnsafe::TVMFFIObjectPtrFromObjectPtr<Object>(data_);
    }
    return AnyView::CopyFromTVMFFIAny(any_data);
  }
};
}  // namespace details

/*!
 * \brief A typed variant container.
 *
 * When all values are ObjectRef, Variant is backed by ObjectRef,
 * otherwise it is backed by Any.
 */
template <typename... V>
class Variant : public details::VariantBase<details::all_object_ref_v<V...>> {
 public:
  using TParent = details::VariantBase<details::all_object_ref_v<V...>>;
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

  Variant(const Variant<V...>& other) : TParent(other.data_) {}
  Variant(Variant<V...>&& other) : TParent(std::move(other.data_)) {}

  TVM_FFI_INLINE Variant& operator=(const Variant<V...>& other) {
    this->SetData(other.data_);
    return *this;
  }

  TVM_FFI_INLINE Variant& operator=(Variant<V...>&& other) {
    this->SetData(std::move(other.data_));
    return *this;
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  Variant(T other) : TParent(std::move(other)) {}  // NOLINT(*)

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE Variant& operator=(T other) {
    return operator=(Variant(std::move(other)));
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE std::optional<T> as() const {
    return this->TParent::ToAnyView().template as<T>();
  }

  /*
   * \brief Shortcut of as Object to cast to a const pointer when T is an Object.
   *
   * \tparam T The object type.
   * \return The requested pointer, returns nullptr if type mismatches.
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
  TVM_FFI_INLINE const T* as() const {
    return this->TParent::ToAnyView().template as<const T*>().value_or(nullptr);
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE T get() const& {
    return this->TParent::ToAnyView().template cast<T>();
  }

  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE T get() && {
    return std::move(*this).TParent::MoveToAny().template cast<T>();
  }

  TVM_FFI_INLINE std::string GetTypeKey() const { return this->TParent::ToAnyView().GetTypeKey(); }

 private:
  friend struct TypeTraits<Variant<V...>>;
  friend struct ObjectPtrHash;
  friend struct ObjectPtrEqual;
  // constructor from any
  explicit Variant(Any data) : TParent(std::move(data)) {}
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
    return this->data_.get();
  }
  // rexpose to friend class
  using TParent::MoveToAny;
  using TParent::ToAnyView;
};

template <typename... V>
inline constexpr bool use_default_type_traits_v<Variant<V...>> = false;

template <typename... V>
struct TypeTraits<Variant<V...>> : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(const Variant<V...>& src, TVMFFIAny* result) {
    *result = src.ToAnyView().CopyToTVMFFIAny();
  }

  TVM_FFI_INLINE static void MoveToAny(Variant<V...> src, TVMFFIAny* result) {
    *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src).MoveToAny());
  }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraitsBase::GetMismatchTypeInfo(src);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return (TypeTraits<V>::CheckAnyStrict(src) || ...);
  }

  TVM_FFI_INLINE static Variant<V...> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return Variant<V...>(Any(AnyView::CopyFromTVMFFIAny(*src)));
  }

  TVM_FFI_INLINE static Variant<V...> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return Variant<V...>(details::AnyUnsafe::MoveTVMFFIAnyToAny(std::move(*src)));
  }

  TVM_FFI_INLINE static std::optional<Variant<V...>> TryCastFromAnyView(const TVMFFIAny* src) {
    // fast path, storage is already in the right type
    if (CheckAnyStrict(src)) {
      return CopyFromAnyViewAfterCheck(src);
    }
    // More expensive path, try to convert to each type, in order of declaration
    return TryVariantTypes<V...>(src);
  }

  template <typename VariantType, typename... Rest>
  TVM_FFI_INLINE static std::optional<Variant<V...>> TryVariantTypes(const TVMFFIAny* src) {
    if (auto opt_convert = TypeTraits<VariantType>::TryCastFromAnyView(src)) {
      return Variant<V...>(*std::move(opt_convert));
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryVariantTypes<Rest...>(src);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return details::ContainerTypeStr<V...>("Variant"); }
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
