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
 * \file tvm/ffi/any.h
 * \brief Any value support.
 */
#ifndef TVM_FFI_ANY_H_
#define TVM_FFI_ANY_H_

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/type_traits.h>

#include <string>
#include <utility>

namespace tvm {
namespace ffi {

class Any;

namespace details {
// Helper to perform
// unsafe operations related to object
struct AnyUnsafe;
}  // namespace details

/*!
 * \brief AnyView allows us to take un-managed reference view of any value.
 */
class AnyView {
 protected:
  /*! \brief The underlying backing data of the any object */
  TVMFFIAny data_;
  // Any can see AnyView
  friend class Any;

 public:
  // NOTE: the following two functions uses styl style
  // since they are common functions appearing in FFI.
  /*!
   * \brief Reset any view to None
   */
  void reset() {
    data_.type_index = TypeIndex::kTVMFFINone;
    // invariance: always set the union padding part to 0
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  TVM_FFI_INLINE void swap(AnyView& other) noexcept { std::swap(data_, other.data_); }
  /*! \return the internal type index */
  TVM_FFI_INLINE int32_t type_index() const noexcept { return data_.type_index; }
  // default constructors
  AnyView() {
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }
  ~AnyView() = default;
  // constructors from any view
  AnyView(const AnyView&) = default;
  AnyView& operator=(const AnyView&) = default;
  AnyView(AnyView&& other) : data_(other.data_) {
    other.data_.type_index = TypeIndex::kTVMFFINone;
    other.data_.zero_padding = 0;
    other.data_.v_int64 = 0;
  }
  TVM_FFI_INLINE AnyView& operator=(AnyView&& other) {
    // copy-and-swap idiom
    AnyView(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  // constructor from general types
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  AnyView(const T& other) {  // NOLINT(*)
    TypeTraits<T>::CopyToAnyView(other, &data_);
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  TVM_FFI_INLINE AnyView& operator=(const T& other) {  // NOLINT(*)
    // copy-and-swap idiom
    AnyView(other).swap(*this);  // NOLINT(*)
    return *this;
  }

  /*!
   * \brief Try to see if we can reinterpret the AnyView to as T object.
   *
   * \tparam T The type to cast to.
   * \return The casted value, or std::nullopt if the cast is not possible.
   * \note This function won't try run type conversion (use try_cast for that purpose).
   */
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  TVM_FFI_INLINE std::optional<T> as() const {
    if (TypeTraits<T>::CheckAnyStrict(&data_)) {
      return TypeTraits<T>::CopyFromAnyViewAfterCheck(&data_);
    } else {
      return std::optional<T>(std::nullopt);
    }
  }
  /*
   * \brief Shortcut of as Object to cast to a const pointer when T is an Object.
   *
   * \tparam T The object type.
   * \return The requested pointer, returns nullptr if type mismatches.
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
  TVM_FFI_INLINE const T* as() const {
    return this->as<const T*>().value_or(nullptr);
  }

  /**
   * \brief Cast to a type T.
   *
   * \tparam T The type to cast to.
   * \return The casted value, or throws an exception if the cast is not possible.
   */
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  TVM_FFI_INLINE T cast() const {
    std::optional<T> opt = TypeTraits<T>::TryCastFromAnyView(&data_);
    if (!opt.has_value()) {
      TVM_FFI_THROW(TypeError) << "Cannot convert from type `"
                               << TypeTraits<T>::GetMismatchTypeInfo(&data_) << "` to `"
                               << TypeTraits<T>::TypeStr() << "`";
    }
    return *std::move(opt);
  }

  /*!
   * \brief Try to cast to a type T, return std::nullopt if the cast is not possible.
   *
   * \tparam T The type to cast to.
   * \return The casted value, or std::nullopt if the cast is not possible.
   */
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  TVM_FFI_INLINE std::optional<T> try_cast() const {
    return TypeTraits<T>::TryCastFromAnyView(&data_);
  }

  // comparison with nullptr
  TVM_FFI_INLINE bool operator==(std::nullptr_t) const noexcept {
    return data_.type_index == TypeIndex::kTVMFFINone;
  }
  TVM_FFI_INLINE bool operator!=(std::nullptr_t) const noexcept {
    return data_.type_index != TypeIndex::kTVMFFINone;
  }
  /*!
   * \brief Get the type key of the Any
   * \return The type key of the Any
   */
  TVM_FFI_INLINE std::string GetTypeKey() const { return TypeIndexToTypeKey(data_.type_index); }
  // The following functions are only used for testing purposes
  /*!
   * \return The underlying supporting data of any view
   * \note This function is used only for testing purposes.
   */
  TVM_FFI_INLINE TVMFFIAny CopyToTVMFFIAny() const { return data_; }
  /*!
   * \return Create an AnyView from TVMFFIAny
   * \param data the underlying ffi data.
   */
  TVM_FFI_INLINE static AnyView CopyFromTVMFFIAny(TVMFFIAny data) {
    AnyView view;
    view.data_ = data;
    return view;
  }
};

namespace details {
/*!
 * \brief Helper function to inplace convert any view to any.
 * \param data The pointer that represents the format as any view.
 * \param extra_any_bytes Indicate that the data may contain extra bytes following
 *  the TVMFFIAny data structure. This is reserved for future possible optimizations
 *  of small-string and extended any object.
 */
TVM_FFI_INLINE void InplaceConvertAnyViewToAny(TVMFFIAny* data,
                                               [[maybe_unused]] size_t extra_any_bytes = 0) {
  if (data->type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin) {
    details::ObjectUnsafe::IncRefObjectHandle(data->v_obj);
  } else if (data->type_index >= TypeIndex::kTVMFFIRawStr) {
    if (data->type_index == TypeIndex::kTVMFFIRawStr) {
      // convert raw string to owned string object
      String temp(data->v_c_str);
      TypeTraits<String>::MoveToAny(std::move(temp), data);
    } else if (data->type_index == TypeIndex::kTVMFFIByteArrayPtr) {
      // convert byte array to owned bytes object
      Bytes temp(*static_cast<TVMFFIByteArray*>(data->v_ptr));
      TypeTraits<Bytes>::MoveToAny(std::move(temp), data);
    } else if (data->type_index == TypeIndex::kTVMFFIObjectRValueRef) {
      // convert rvalue ref to owned object
      Object** obj_addr = static_cast<Object**>(data->v_ptr);
      TVM_FFI_ICHECK(obj_addr[0] != nullptr) << "RValueRef already moved";
      ObjectRef temp(details::ObjectUnsafe::ObjectPtrFromOwned<Object>(obj_addr[0]));
      // set the rvalue ref to nullptr to avoid double move
      obj_addr[0] = nullptr;
      TypeTraits<ObjectRef>::MoveToAny(std::move(temp), data);
    }
  }
}
}  // namespace details

/*!
 * \brief Managed Any that takes strong reference to a value.
 *
 * \note Develooper invariance: the TVMFFIAny data_
 *       in the Any can be safely used in AnyView.
 */
class Any {
 protected:
  /*! \brief The underlying backing data of the any object */
  TVMFFIAny data_;

 public:
  /*!
   * \brief Reset any to None
   */
  TVM_FFI_INLINE void reset() {
    if (data_.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectUnsafe::DecRefObjectHandle(data_.v_obj);
    }
    data_.type_index = TVMFFITypeIndex::kTVMFFINone;
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  TVM_FFI_INLINE void swap(Any& other) noexcept { std::swap(data_, other.data_); }
  /*! \return the internal type index */
  TVM_FFI_INLINE int32_t type_index() const noexcept { return data_.type_index; }
  // default constructors
  Any() {
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }
  ~Any() { this->reset(); }
  // constructors from Any
  Any(const Any& other) : data_(other.data_) {
    if (data_.type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectUnsafe::IncRefObjectHandle(data_.v_obj);
    }
  }
  Any(Any&& other) : data_(other.data_) {
    other.data_.type_index = TypeIndex::kTVMFFINone;
    other.data_.zero_padding = 0;
    other.data_.v_int64 = 0;
  }
  TVM_FFI_INLINE Any& operator=(const Any& other) {
    // copy-and-swap idiom
    Any(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  TVM_FFI_INLINE Any& operator=(Any&& other) {
    // copy-and-swap idiom
    Any(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  // convert from/to AnyView
  Any(const AnyView& other) : data_(other.data_) {  // NOLINT(*)
    details::InplaceConvertAnyViewToAny(&data_);
  }
  TVM_FFI_INLINE Any& operator=(const AnyView& other) {
    // copy-and-swap idiom
    Any(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*! \brief Any can be converted to AnyView in zero cost. */
  operator AnyView() const { return AnyView::CopyFromTVMFFIAny(data_); }
  // constructor from general types
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  Any(T other) {  // NOLINT(*)
    TypeTraits<T>::MoveToAny(std::move(other), &data_);
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  TVM_FFI_INLINE Any& operator=(T other) {  // NOLINT(*)
    // copy-and-swap idiom
    Any(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }

  /**
   * \brief Try to reinterpret the Any as a type T, return std::nullopt if it is not possible.
   *
   * \tparam T The type to cast to.
   * \return The casted value, or std::nullopt if the cast is not possible.
   * \note This function won't try to run type conversion (use try_cast for that purpose).
   */
  template <typename T,
            typename = std::enable_if_t<TypeTraits<T>::storage_enabled || std::is_same_v<T, Any>>>
  TVM_FFI_INLINE std::optional<T> as() && {
    if constexpr (std::is_same_v<T, Any>) {
      return std::move(*this);
    } else {
      if (TypeTraits<T>::CheckAnyStrict(&data_)) {
        return TypeTraits<T>::MoveFromAnyAfterCheck(&data_);
      } else {
        return std::optional<T>(std::nullopt);
      }
    }
  }

  /**
   * \brief Try to reinterpret the Any as a type T, return std::nullopt if it is not possible.
   *
   * \tparam T The type to cast to.
   * \return The casted value, or std::nullopt if the cast is not possible.
   * \note This function won't try to run type conversion (use try_cast for that purpose).
   */
  template <typename T,
            typename = std::enable_if_t<TypeTraits<T>::convert_enabled || std::is_same_v<T, Any>>>
  TVM_FFI_INLINE std::optional<T> as() const& {
    if constexpr (std::is_same_v<T, Any>) {
      return *this;
    } else {
      if (TypeTraits<T>::CheckAnyStrict(&data_)) {
        return TypeTraits<T>::CopyFromAnyViewAfterCheck(&data_);
      } else {
        return std::optional<T>(std::nullopt);
      }
    }
  }

  /*
   * \brief Shortcut of as Object to cast to a const pointer when T is an Object.
   *
   * \tparam T The object type.
   * \return The requested pointer, returns nullptr if type mismatches.
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
  TVM_FFI_INLINE const T* as() const& {
    return this->as<const T*>().value_or(nullptr);
  }

  /**
   * \brief Cast to a type T, throw an exception if the cast is not possible.
   *
   * \tparam T The type to cast to.
   */
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::convert_enabled>>
  TVM_FFI_INLINE T cast() const& {
    std::optional<T> opt = TypeTraits<T>::TryCastFromAnyView(&data_);
    if (!opt.has_value()) {
      TVM_FFI_THROW(TypeError) << "Cannot convert from type `"
                               << TypeTraits<T>::GetMismatchTypeInfo(&data_) << "` to `"
                               << TypeTraits<T>::TypeStr() << "`";
    }
    return *std::move(opt);
  }

  /**
   * \brief Cast to a type T, throw an exception if the cast is not possible.
   *
   * \tparam T The type to cast to.
   */
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::storage_enabled>>
  TVM_FFI_INLINE T cast() && {
    if (TypeTraits<T>::CheckAnyStrict(&data_)) {
      return TypeTraits<T>::MoveFromAnyAfterCheck(&data_);
    }
    // slow path, try to do fallback convert
    std::optional<T> opt = TypeTraits<T>::TryCastFromAnyView(&data_);
    if (!opt.has_value()) {
      TVM_FFI_THROW(TypeError) << "Cannot convert from type `"
                               << TypeTraits<T>::GetMismatchTypeInfo(&data_) << "` to `"
                               << TypeTraits<T>::TypeStr() << "`";
    }
    return *std::move(opt);
  }

  /**
   * \brief Try to cast to a type T.
   *
   * \tparam T The type to cast to.
   * \return The casted value, or std::nullopt if the cast is not possible.
   * \note use STL name since it to be more consistent with cast API.
   */
  template <typename T,
            typename = std::enable_if_t<TypeTraits<T>::convert_enabled || std::is_same_v<T, Any>>>
  TVM_FFI_INLINE std::optional<T> try_cast() const {
    if constexpr (std::is_same_v<T, Any>) {
      return *this;
    } else {
      return TypeTraits<T>::TryCastFromAnyView(&data_);
    }
  }
  /*
   * \brief Check if the two Any are same type and value in shallow comparison.
   * \param other The other Any
   * \return True if the two Any are same type and value, false otherwise.
   */
  TVM_FFI_INLINE bool same_as(const Any& other) const noexcept {
    return data_.type_index == other.data_.type_index &&
           data_.zero_padding == other.data_.zero_padding && data_.v_int64 == other.data_.v_int64;
  }

  /*
   * \brief Check if any and ObjectRef are same type and value in shallow comparison.
   * \param other The other ObjectRef
   * \return True if the two Any are same type and value, false otherwise.
   */
  TVM_FFI_INLINE bool same_as(const ObjectRef& other) const noexcept {
    if (other.get() != nullptr) {
      return (data_.type_index == other->type_index() &&
              reinterpret_cast<Object*>(data_.v_obj) == other.get());
    } else {
      return data_.type_index == TypeIndex::kTVMFFINone;
    }
  }

  TVM_FFI_INLINE bool operator==(std::nullptr_t) const noexcept {
    return data_.type_index == TypeIndex::kTVMFFINone;
  }
  TVM_FFI_INLINE bool operator!=(std::nullptr_t) const noexcept {
    return data_.type_index != TypeIndex::kTVMFFINone;
  }

  /*!
   * \brief Get the type key of the Any
   * \return The type key of the Any
   */
  TVM_FFI_INLINE std::string GetTypeKey() const { return TypeIndexToTypeKey(data_.type_index); }

  friend struct details::AnyUnsafe;
  friend struct AnyHash;
  friend struct AnyEqual;
};

// layout assert to ensure we can freely cast between the two types
static_assert(sizeof(AnyView) == sizeof(TVMFFIAny));
static_assert(sizeof(Any) == sizeof(TVMFFIAny));

namespace details {

template <typename Type>
struct Type2Str {
  static std::string v() { return TypeTraitsNoCR<Type>::TypeStr(); }
};

template <>
struct Type2Str<Any> {
  static std::string v() { return "Any"; }
};

template <>
struct Type2Str<const Any&> {
  static std::string v() { return "Any"; }
};

template <>
struct Type2Str<AnyView> {
  static std::string v() { return "AnyView"; }
};

template <>
struct Type2Str<const AnyView&> {
  static std::string v() { return "AnyView"; }
};

template <>
struct Type2Str<void> {
  static std::string v() { return "void"; }
};

// Extra unsafe method to help any manipulation
struct AnyUnsafe : public ObjectUnsafe {
  // FFI related operations
  TVM_FFI_INLINE static TVMFFIAny MoveAnyToTVMFFIAny(Any&& any) {
    TVMFFIAny result = any.data_;
    any.data_.type_index = TypeIndex::kTVMFFINone;
    any.data_.zero_padding = 0;
    any.data_.v_int64 = 0;
    return result;
  }

  TVM_FFI_INLINE static Any MoveTVMFFIAnyToAny(TVMFFIAny&& data) {
    Any any;
    any.data_ = data;
    data.type_index = TypeIndex::kTVMFFINone;
    data.zero_padding = 0;
    data.v_int64 = 0;
    return any;
  }

  template <typename T>
  TVM_FFI_INLINE static bool CheckAnyStrict(const Any& ref) {
    return TypeTraits<T>::CheckAnyStrict(&(ref.data_));
  }

  template <typename T>
  TVM_FFI_INLINE static T CopyFromAnyViewAfterCheck(const Any& ref) {
    if constexpr (!std::is_same_v<T, Any>) {
      return TypeTraits<T>::CopyFromAnyViewAfterCheck(&(ref.data_));
    } else {
      return ref;
    }
  }

  template <typename T>
  TVM_FFI_INLINE static T MoveFromAnyAfterCheck(Any&& ref) {
    if constexpr (!std::is_same_v<T, Any>) {
      return TypeTraits<T>::MoveFromAnyAfterCheck(&(ref.data_));
    } else {
      return std::move(ref);
    }
  }

  TVM_FFI_INLINE static Object* ObjectPtrFromAnyAfterCheck(const Any& ref) {
    return reinterpret_cast<Object*>(ref.data_.v_obj);
  }

  TVM_FFI_INLINE static const TVMFFIAny* TVMFFIAnyPtrFromAny(const Any& ref) {
    return &(ref.data_);
  }

  template <typename T>
  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const Any& ref) {
    return TypeTraits<T>::GetMismatchTypeInfo(&(ref.data_));
  }
};
}  // namespace details

/*! \brief String-aware Any equal functor */
struct AnyHash {
  /*!
   * \brief Calculate the hash code of an Any
   * \param a The given Any
   * \return Hash code of a, string hash for strings and pointer address otherwise.
   */
  uint64_t operator()(const Any& src) const {
    if (src.data_.type_index == TypeIndex::kTVMFFISmallStr) {
      // for small string, we use the same type key hash as normal string
      // so heap allocated string and on stack string will have the same hash
      return details::StableHashCombine(TypeIndex::kTVMFFIStr,
                                        details::StableHashSmallStrBytes(&src.data_));
    } else if (src.data_.type_index == TypeIndex::kTVMFFISmallBytes) {
      // use byte the same type key as bytes
      return details::StableHashCombine(TypeIndex::kTVMFFIBytes,
                                        details::StableHashSmallStrBytes(&src.data_));
    } else if (src.data_.type_index == TypeIndex::kTVMFFIStr ||
               src.data_.type_index == TypeIndex::kTVMFFIBytes) {
      const details::BytesObjBase* src_str =
          details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(src);
      return details::StableHashCombine(src.data_.type_index,
                                        details::StableHashBytes(src_str->data, src_str->size));
    } else {
      return details::StableHashCombine(src.data_.type_index, src.data_.v_uint64);
    }
  }
};

/*! \brief String-aware Any hash functor */
struct AnyEqual {
  /*!
   * \brief Check if the two Any are equal
   * \param lhs left operand.
   * \param rhs right operand
   * \return String equality if both are strings, pointer address equality otherwise.
   */
  bool operator()(const Any& lhs, const Any& rhs) const {
    // header with type index
    const int64_t* lhs_as_int64 = reinterpret_cast<const int64_t*>(&lhs.data_);
    const int64_t* rhs_as_int64 = reinterpret_cast<const int64_t*>(&rhs.data_);
    static_assert(sizeof(TVMFFIAny) == 16);
    // fast path, check byte equality
    if (lhs_as_int64[0] == rhs_as_int64[0] && lhs_as_int64[1] == rhs_as_int64[1]) {
      return true;
    }
    // common false case type index match, in this case we only need to pay attention to string
    // equality
    if (lhs.data_.type_index == rhs.data_.type_index) {
      // specialy handle string hash
      if (lhs.data_.type_index == TypeIndex::kTVMFFIStr ||
          lhs.data_.type_index == TypeIndex::kTVMFFIBytes) {
        const details::BytesObjBase* lhs_str =
            details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(lhs);
        const details::BytesObjBase* rhs_str =
            details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(rhs);
        return Bytes::memequal(lhs_str->data, rhs_str->data, lhs_str->size, rhs_str->size);
      }
      return false;
    } else {
      // type_index mismatch, if index is not string, return false
      if (lhs.data_.type_index != kTVMFFIStr && lhs.data_.type_index != kTVMFFISmallStr &&
          lhs.data_.type_index != kTVMFFISmallBytes && lhs.data_.type_index != kTVMFFIBytes) {
        return false;
      }
      // small string and normal string comparison
      if (lhs.data_.type_index == kTVMFFIStr && rhs.data_.type_index == kTVMFFISmallStr) {
        const details::BytesObjBase* lhs_str =
            details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(lhs);
        return Bytes::memequal(lhs_str->data, rhs.data_.v_bytes, lhs_str->size,
                               rhs.data_.small_str_len);
      }
      if (lhs.data_.type_index == kTVMFFISmallStr && rhs.data_.type_index == kTVMFFIStr) {
        const details::BytesObjBase* rhs_str =
            details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(rhs);
        return Bytes::memequal(lhs.data_.v_bytes, rhs_str->data, lhs.data_.small_str_len,
                               rhs_str->size);
      }
      if (lhs.data_.type_index == kTVMFFIBytes && rhs.data_.type_index == kTVMFFISmallBytes) {
        const details::BytesObjBase* lhs_bytes =
            details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(lhs);
        return Bytes::memequal(lhs_bytes->data, rhs.data_.v_bytes, lhs_bytes->size,
                               rhs.data_.small_str_len);
      }
      if (lhs.data_.type_index == kTVMFFISmallBytes && rhs.data_.type_index == kTVMFFIBytes) {
        const details::BytesObjBase* rhs_bytes =
            details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(rhs);
        return Bytes::memequal(lhs.data_.v_bytes, rhs_bytes->data, lhs.data_.small_str_len,
                               rhs_bytes->size);
      }
      return false;
    }
  }
};
}  // namespace ffi

// Expose to the tvm namespace for usability
// Rationale: no ambiguity even in root
using tvm::ffi::Any;
using tvm::ffi::AnyView;

}  // namespace tvm
#endif  // TVM_FFI_ANY_H_
