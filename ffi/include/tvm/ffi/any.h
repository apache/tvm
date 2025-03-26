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
    data_.v_int64 = 0;
  }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(AnyView& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*! \return the internal type index */
  int32_t type_index() const { return data_.type_index; }
  // default constructors
  AnyView() {
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.v_int64 = 0;
  }
  ~AnyView() = default;
  // constructors from any view
  AnyView(const AnyView&) = default;
  AnyView& operator=(const AnyView&) = default;
  AnyView(AnyView&& other) : data_(other.data_) {
    other.data_.type_index = TypeIndex::kTVMFFINone;
    other.data_.v_int64 = 0;
  }
  AnyView& operator=(AnyView&& other) {
    // copy-and-swap idiom
    AnyView(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  // constructor from general types
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  AnyView(const T& other) {  // NOLINT(*)
    TypeTraits<T>::CopyToAnyView(other, &data_);
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  AnyView& operator=(const T& other) {  // NOLINT(*)
    // copy-and-swap idiom
    AnyView(other).swap(*this);  // NOLINT(*)
    return *this;
  }

  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  std::optional<T> as() const {
    return TypeTraits<T>::TryCopyFromAnyView(&data_);
  }

  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  operator T() const {
    std::optional<T> opt = TypeTraits<T>::TryCopyFromAnyView(&data_);
    if (opt.has_value()) {
      return std::move(*opt);
    }
    TVM_FFI_THROW(TypeError) << "Cannot convert from type `"
                             << TypeTraits<T>::GetMismatchTypeInfo(&data_) << "` to `"
                             << TypeTraits<T>::TypeStr() << "`";
    TVM_FFI_UNREACHABLE();
  }
  // The following functions are only used for testing purposes
  /*!
   * \return The underlying supporting data of any view
   * \note This function is used only for testing purposes.
   */
  TVMFFIAny CopyToTVMFFIAny() const { return data_; }
  /*!
   * \return Create an AnyView from TVMFFIAny
   * \param data the underlying ffi data.
   */
  static AnyView CopyFromTVMFFIAny(TVMFFIAny data) {
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
    details::ObjectUnsafe::IncRefObjectInAny(data);
  } else if (data->type_index == TypeIndex::kTVMFFIRawStr) {
    // convert raw string to owned string object
    String temp(data->v_c_str);
    data->type_index = TypeIndex::kTVMFFIStr;
    data->v_obj = details::ObjectUnsafe::MoveTVMFFIObjectPtrFromObjectRef(&temp);
  } else if (data->type_index == TypeIndex::kTVMFFIByteArrayPtr) {
    // convert byte array to owned bytes object
    Bytes temp(*static_cast<TVMFFIByteArray*>(data->v_ptr));
    data->type_index = TypeIndex::kTVMFFIBytes;
    data->v_obj = details::ObjectUnsafe::MoveTVMFFIObjectPtrFromObjectRef(&temp);
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
  void reset() {
    if (data_.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectUnsafe::DecRefObjectInAny(&data_);
    }
    data_.type_index = TVMFFITypeIndex::kTVMFFINone;
    data_.v_int64 = 0;
  }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(Any& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*! \return the internal type index */
  int32_t type_index() const { return data_.type_index; }
  // default constructors
  Any() {
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.v_int64 = 0;
  }
  ~Any() { this->reset(); }
  // constructors from Any
  Any(const Any& other) : data_(other.data_) {
    if (data_.type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectUnsafe::IncRefObjectInAny(&data_);
    }
  }
  Any(Any&& other) : data_(other.data_) {
    other.data_.type_index = TypeIndex::kTVMFFINone;
    other.data_.v_int64 = 0;
  }
  Any& operator=(const Any& other) {
    // copy-and-swap idiom
    Any(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  Any& operator=(Any&& other) {
    // copy-and-swap idiom
    Any(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  // convert from/to AnyView
  Any(const AnyView& other) : data_(other.data_) {  // NOLINT(*)
    details::InplaceConvertAnyViewToAny(&data_);
  }
  Any& operator=(const AnyView& other) {
    // copy-and-swap idiom
    Any(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*! \brief Any can be converted to AnyView in zero cost. */
  operator AnyView() const { return AnyView::CopyFromTVMFFIAny(data_); }
  // constructor from general types
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  Any(T other) {  // NOLINT(*)
    TypeTraits<T>::MoveToAny(std::move(other), &data_);
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  Any& operator=(T other) {  // NOLINT(*)
    // copy-and-swap idiom
    Any(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  std::optional<T> as() const {
    return TypeTraits<T>::TryCopyFromAnyView(&data_);
  }

  /*
   * \brief Shortcut of as Object to cast to a const pointer when T is an Object.
   *
   * \tparam T The object type.
   * \return The requested pointer, returns nullptr if type mismatches.
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
  const T* as() const {
    return this->as<const T*>().value_or(nullptr);
  }

  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  operator T() const {
    std::optional<T> opt = TypeTraits<T>::TryCopyFromAnyView(&data_);
    if (opt.has_value()) {
      return std::move(*opt);
    }
    TVM_FFI_THROW(TypeError) << "Cannot convert from type `"
                             << TypeTraits<T>::GetMismatchTypeInfo(&data_) << "` to `"
                             << TypeTraits<T>::TypeStr() << "`";
    TVM_FFI_UNREACHABLE();
  }

  /*
   * \brief Check if the two Any are same type and value in shallow comparison.
   * \param other The other Any
   * \return True if the two Any are same type and value, false otherwise.
   */
  bool same_as(const Any& other) const {
    return data_.type_index == other.data_.type_index && data_.v_int64 == other.data_.v_int64;
  }

  bool operator==(std::nullptr_t) const { return data_.type_index == TypeIndex::kTVMFFINone; }

  bool operator!=(std::nullptr_t) const { return data_.type_index != TypeIndex::kTVMFFINone; }

  // FFI related operations
  /*!
   * Move the current data to FFI any
   * \param result the output to nmove to
   */
  void MoveToTVMFFIAny(TVMFFIAny* result) {
    *result = data_;
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.v_int64 = 0;
  }

  /*!
   * \brief Move the current data to FFI any
   * \param data the input to move from
   */
  static Any MoveFromTVMFFIAny(TVMFFIAny data) {
    Any any;
    any.data_ = data;
    return any;
  }

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
  /*!
   * \brief Internal helper function downcast a any that already passes check.
   * \note Only used for internal dev purposes.
   * \tparam T The target reference type.
   * \return The casted result.
   */
  template <typename T>
  static TVM_FFI_INLINE T ConvertAfterCheck(const Any& ref) {
    if constexpr (!std::is_same_v<T, Any>) {
      return TypeTraits<T>::CopyFromAnyViewAfterCheck(&(ref.data_));
    } else {
      return ref;
    }
  }
  template <typename T>
  static TVM_FFI_INLINE bool CheckAny(const Any& ref) {
    return TypeTraits<T>::CheckAnyView(&(ref.data_));
  }

  static TVM_FFI_INLINE Object* GetObjectPtrFromAny(const Any& ref) {
    return reinterpret_cast<Object*>(ref.data_.v_obj);
  }

  static TVM_FFI_INLINE const TVMFFIAny* GetTVMFFIAnyPtrFromAny(const Any& ref) {
    return &(ref.data_);
  }
  template <typename T>
  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const Any& ref) {
    return TypeTraits<T>::GetMismatchTypeInfo(&(ref.data_));
  }
};
}  // namespace details

/*! \brief String-aware ObjectRef equal functor */
struct AnyHash {
  /*!
   * \brief Calculate the hash code of an Any
   * \param a The given Any
   * \return Hash code of a, string hash for strings and pointer address otherwise.
   */
  uint64_t operator()(const Any& src) const {
    uint64_t val_hash = [&]() -> uint64_t {
      if (src.data_.type_index == TypeIndex::kTVMFFIStr ||
          src.data_.type_index == TypeIndex::kTVMFFIBytes) {
        const BytesObjBase* src_str =
            details::AnyUnsafe::ConvertAfterCheck<const BytesObjBase*>(src);
        return details::StableHashBytes(src_str->bytes.data, src_str->bytes.size);
      } else {
        return src.data_.v_uint64;
      }
    }();
    return details::StableHashCombine(src.data_.type_index, val_hash);
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
    if (lhs.data_.type_index != rhs.data_.type_index) return false;
    // byte equivalence
    if (lhs.data_.v_int64 == rhs.data_.v_int64) return true;
    // specialy handle string hash
    if (lhs.data_.type_index == TypeIndex::kTVMFFIStr ||
        lhs.data_.type_index == TypeIndex::kTVMFFIBytes) {
      const BytesObjBase* lhs_str = details::AnyUnsafe::ConvertAfterCheck<const BytesObjBase*>(lhs);
      const BytesObjBase* rhs_str = details::AnyUnsafe::ConvertAfterCheck<const BytesObjBase*>(rhs);
      return Bytes::memncmp(lhs_str->bytes.data, rhs_str->bytes.data, lhs_str->bytes.size,
                            rhs_str->bytes.size) == 0;
    }
    return false;
  }
};
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_ANY_H_
