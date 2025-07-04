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
 * \file tvm/ffi/string.h
 * \brief Runtime Bytes and String types.
 */
#ifndef TVM_FFI_STRING_H_
#define TVM_FFI_STRING_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// Note: We place string in tvm/ffi instead of tvm/ffi/container
// because string itself needs special handling and is an inherent
// core component for return string handling.
// The following dependency relation holds
// any -> string -> object

namespace tvm {
namespace ffi {

/*! \brief Base class for bytes and string. */
class BytesObjBase : public Object, public TVMFFIByteArray {};

/*!
 * \brief An object representing bytes.
 * \note We use separate object for bytes to follow python convention
 *       and indicate passing of raw bytes.
 *       Bytes can be converted from/to string.
 */
class BytesObj : public BytesObjBase {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kTVMFFIBytes;
  static constexpr const char* _type_key = StaticTypeKey::kTVMFFIBytes;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_STATIC_OBJECT_INFO(BytesObj, Object);
};

/*! \brief An object representing string. It's POD type. */
class StringObj : public BytesObjBase {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kTVMFFIStr;
  static constexpr const char* _type_key = StaticTypeKey::kTVMFFIStr;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_STATIC_OBJECT_INFO(StringObj, Object);
};

namespace details {

// String moved from std::string
// without having to trigger a copy
template <typename Base>
class BytesObjStdImpl : public Base {
 public:
  explicit BytesObjStdImpl(std::string other) : data_{other} {
    this->data = data_.data();
    this->size = data_.size();
  }

 private:
  std::string data_;
};

// inplace string allocation
template <typename Base>
TVM_FFI_INLINE ObjectPtr<Base> MakeInplaceBytes(const char* data, size_t length) {
  ObjectPtr<Base> p = make_inplace_array_object<Base, char>(length + 1);
  static_assert(alignof(Base) % alignof(char) == 0);
  static_assert(sizeof(Base) % alignof(char) == 0);
  char* dest_data = reinterpret_cast<char*>(p.get()) + sizeof(Base);
  p->data = dest_data;
  p->size = length;
  std::memcpy(dest_data, data, length);
  dest_data[length] = '\0';
  return p;
}
}  // namespace details

/*!
 * \brief Managed reference of byte array.
 */
class Bytes : public ObjectRef {
 public:
  /*!
   * \brief constructor from char [N]
   *
   * \param other a char array.
   */
  Bytes(const char* data, size_t size)  // NOLINT(*)
      : ObjectRef(details::MakeInplaceBytes<BytesObj>(data, size)) {}
  /*!
   * \brief constructor from char [N]
   *
   * \param other a char array.
   */
  Bytes(TVMFFIByteArray bytes)  // NOLINT(*)
      : ObjectRef(details::MakeInplaceBytes<BytesObj>(bytes.data, bytes.size)) {}
  /*!
   * \brief constructor from char [N]
   *
   * \param other a char array.
   */
  Bytes(std::string other)  // NOLINT(*)
      : ObjectRef(make_object<details::BytesObjStdImpl<BytesObj>>(std::move(other))) {}
  /*!
   * \brief Swap this String with another string
   * \param other The other string
   */
  void swap(Bytes& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }

  template <typename T>
  Bytes& operator=(T&& other) {
    // copy-and-swap idiom
    Bytes(std::forward<T>(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t size() const { return get()->size; }
  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  const char* data() const { return get()->data; }
  /*!
   * \brief Convert String to an std::string object
   *
   * \return std::string
   */
  operator std::string() const { return std::string{get()->data, size()}; }

  TVM_FFI_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Bytes, ObjectRef, BytesObj);

 private:
  /*!
   * \brief Compare two char sequence
   *
   * \param lhs Pointers to the char array to compare
   * \param rhs Pointers to the char array to compare
   * \param lhs_count Length of the char array to compare
   * \param rhs_count Length of the char array to compare
   * \return int zero if both char sequences compare equal. negative if this
   * appear before other, positive otherwise.
   */
  static int memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count);

  friend struct AnyEqual;
  friend class String;
};

/*!
 * \brief Reference to string objects.
 *
 * \code
 *
 * // Example to create runtime String reference object from std::string
 * std::string s = "hello world";
 *
 * // You can create the reference from existing std::string
 * String ref{std::move(s)};
 *
 * // You can rebind the reference to another string.
 * ref = std::string{"hello world2"};
 *
 * // You can use the reference as hash map key
 * std::unordered_map<String, int32_t> m;
 * m[ref] = 1;
 *
 * // You can compare the reference object with other string objects
 * assert(ref == "hello world", true);
 *
 * // You can convert the reference to std::string again
 * string s2 = (string)ref;
 *
 * \endcode
 */
class String : public ObjectRef {
 public:
  String(std::nullptr_t) = delete;  // NOLINT(*)

  /*!
   * \brief constructor from char [N]
   *
   * \param other a char array.
   */
  template <size_t N>
  String(const char other[N])  // NOLINT(*)
      : ObjectRef(details::MakeInplaceBytes<StringObj>(other, N)) {}

  /*!
   * \brief constructor
   */
  String() : String("") {}

  /*!
   * \brief constructor from raw string
   *
   * \param other a char array.
   */
  String(const char* other)  // NOLINT(*)
      : ObjectRef(details::MakeInplaceBytes<StringObj>(other, std::strlen(other))) {}

  /*!
   * \brief constructor from raw string
   *
   * \param other a char array.
   */
  String(const char* other, size_t size)  // NOLINT(*)
      : ObjectRef(details::MakeInplaceBytes<StringObj>(other, size)) {}

  /*!
   * \brief Construct a new string object
   * \param other The std::string object to be copied
   */
  String(const std::string& other)  // NOLINT(*)
      : ObjectRef(details::MakeInplaceBytes<StringObj>(other.data(), other.size())) {}

  /*!
   * \brief Construct a new string object
   * \param other The std::string object to be moved
   */
  String(std::string&& other)  // NOLINT(*)
      : ObjectRef(make_object<details::BytesObjStdImpl<StringObj>>(std::move(other))) {}

  /*!
   * \brief constructor from TVMFFIByteArray
   *
   * \param other a TVMFFIByteArray.
   */
  explicit String(TVMFFIByteArray other)
      : ObjectRef(details::MakeInplaceBytes<StringObj>(other.data, other.size)) {}

  /*!
   * \brief Swap this String with another string
   * \param other The other string
   */
  void swap(String& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }

  template <typename T>
  String& operator=(T&& other) {
    // copy-and-swap idiom
    String(std::forward<T>(other)).swap(*this);  // NOLINT(*)
    return *this;
  }

  /*!
   * \brief Compares this String object to other
   *
   * \param other The String to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const String& other) const {
    return Bytes::memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this String object to other
   *
   * \param other The string to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const std::string& other) const {
    return Bytes::memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this to other
   *
   * \param other The character array to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const char* other) const {
    return Bytes::memncmp(data(), other, size(), std::strlen(other));
  }

  /*!
   * \brief Compares this to other
   *
   * \param other The TVMFFIByteArray to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const TVMFFIByteArray& other) const {
    return Bytes::memncmp(data(), other.data, size(), other.size);
  }

  /*!
   * \brief Returns a pointer to the char array in the string.
   *
   * \return const char*
   */
  const char* c_str() const { return get()->data; }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t size() const {
    const auto* ptr = get();
    return ptr->size;
  }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t length() const { return size(); }

  /*!
   * \brief Retun if the string is empty
   *
   * \return true if empty, false otherwise.
   */
  bool empty() const { return size() == 0; }

  /*!
   * \brief Read an element.
   * \param pos The position at which to read the character.
   *
   * \return The char at position
   */
  char at(size_t pos) const {
    if (pos < size()) {
      return data()[pos];
    } else {
      throw std::out_of_range("tvm::String index out of bounds");
    }
  }

  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  const char* data() const { return get()->data; }

  /*!
   * \brief Convert String to an std::string object
   *
   * \return std::string
   */
  operator std::string() const { return std::string{get()->data, size()}; }

  TVM_FFI_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(String, ObjectRef, StringObj);

 private:
  /*!
   * \brief Concatenate two char sequences
   *
   * \param lhs Pointers to the lhs char array
   * \param lhs_size The size of the lhs char array
   * \param rhs Pointers to the rhs char array
   * \param rhs_size The size of the rhs char array
   *
   * \return The concatenated char sequence
   */
  static String Concat(const char* lhs, size_t lhs_size, const char* rhs, size_t rhs_size) {
    std::string ret(lhs, lhs_size);
    ret.append(rhs, rhs_size);
    return String(ret);
  }

  // Overload + operator
  friend String operator+(const String& lhs, const String& rhs);
  friend String operator+(const String& lhs, const std::string& rhs);
  friend String operator+(const std::string& lhs, const String& rhs);
  friend String operator+(const String& lhs, const char* rhs);
  friend String operator+(const char* lhs, const String& rhs);
};

/*! \brief Convert TVMFFIByteArray to std::string_view */
TVM_FFI_INLINE std::string_view ToStringView(TVMFFIByteArray str) {
  return std::string_view(str.data, str.size);
}

// const char*, requirement: not nullable, do not retain ownership
template <int N>
struct TypeTraits<char[N]> : public TypeTraitsBase {
  // NOTE: only enable implicit conversion into AnyView
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static void CopyToAnyView(const char src[N], TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIRawStr;
    result->v_c_str = src;
  }

  TVM_FFI_INLINE static void MoveToAny(const char src[N], TVMFFIAny* result) {
    // when we need to move to any, convert to owned object first
    ObjectRefTypeTraitsBase<String>::MoveToAny(String(src), result);
  }
};

template <>
struct TypeTraits<const char*> : public TypeTraitsBase {
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static void CopyToAnyView(const char* src, TVMFFIAny* result) {
    TVM_FFI_ICHECK_NOTNULL(src);
    result->type_index = TypeIndex::kTVMFFIRawStr;
    result->v_c_str = src;
  }

  TVM_FFI_INLINE static void MoveToAny(const char* src, TVMFFIAny* result) {
    // when we need to move to any, convert to owned object first
    ObjectRefTypeTraitsBase<String>::MoveToAny(String(src), result);
  }
  // Do not allow const char* in a container, so we do not need CheckAnyStrict
  TVM_FFI_INLINE static std::optional<const char*> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIRawStr) {
      return static_cast<const char*>(src->v_c_str);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "const char*"; }
};

// TVMFFIByteArray, requirement: not nullable, do not retain ownership
template <>
struct TypeTraits<TVMFFIByteArray*> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIByteArrayPtr;
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static void CopyToAnyView(TVMFFIByteArray* src, TVMFFIAny* result) {
    TVM_FFI_ICHECK_NOTNULL(src);
    result->type_index = TypeIndex::kTVMFFIByteArrayPtr;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_ptr = src;
  }

  TVM_FFI_INLINE static void MoveToAny(TVMFFIByteArray* src, TVMFFIAny* result) {
    ObjectRefTypeTraitsBase<Bytes>::MoveToAny(Bytes(*src), result);
  }

  TVM_FFI_INLINE static std::optional<TVMFFIByteArray*> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIByteArrayPtr) {
      return static_cast<TVMFFIByteArray*>(src->v_ptr);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIByteArrayPtr; }
};

template <>
inline constexpr bool use_default_type_traits_v<Bytes> = false;

// specialize to enable implicit conversion from TVMFFIByteArray*
template <>
struct TypeTraits<Bytes> : public ObjectRefWithFallbackTraitsBase<Bytes, TVMFFIByteArray*> {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBytes;
  TVM_FFI_INLINE static Bytes ConvertFallbackValue(TVMFFIByteArray* src) { return Bytes(*src); }
};

template <>
inline constexpr bool use_default_type_traits_v<String> = false;

// specialize to enable implicit conversion from const char*
template <>
struct TypeTraits<String> : public ObjectRefWithFallbackTraitsBase<String, const char*> {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIStr;
  TVM_FFI_INLINE static String ConvertFallbackValue(const char* src) { return String(src); }
};

template <>
inline constexpr bool use_default_type_traits_v<std::string> = false;

template <>
struct TypeTraits<std::string>
    : public FallbackOnlyTraitsBase<std::string, const char*, TVMFFIByteArray*, Bytes, String> {
  TVM_FFI_INLINE static void CopyToAnyView(const std::string& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIRawStr;
    result->v_c_str = src.c_str();
  }

  TVM_FFI_INLINE static void MoveToAny(std::string src, TVMFFIAny* result) {
    // when we need to move to any, convert to owned object first
    ObjectRefTypeTraitsBase<String>::MoveToAny(String(std::move(src)), result);
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "std::string"; }

  TVM_FFI_INLINE static std::string ConvertFallbackValue(const char* src) {
    return std::string(src);
  }

  TVM_FFI_INLINE static std::string ConvertFallbackValue(TVMFFIByteArray* src) {
    return std::string(src->data, src->size);
  }

  TVM_FFI_INLINE static std::string ConvertFallbackValue(Bytes src) {
    return src.operator std::string();
  }

  TVM_FFI_INLINE static std::string ConvertFallbackValue(String src) {
    return src.operator std::string();
  }
};

inline String operator+(const String& lhs, const String& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const String& lhs, const std::string& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const std::string& lhs, const String& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const char* lhs, const String& rhs) {
  size_t lhs_size = std::strlen(lhs);
  size_t rhs_size = rhs.size();
  return String::Concat(lhs, lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const String& lhs, const char* rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = std::strlen(rhs);
  return String::Concat(lhs.data(), lhs_size, rhs, rhs_size);
}

// Overload < operator
inline bool operator<(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

inline bool operator<(const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const String& lhs, const char* rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const char* lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

// Overload > operator
inline bool operator>(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

inline bool operator>(const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const String& lhs, const char* rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const char* lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

// Overload <= operator
inline bool operator<=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

inline bool operator<=(const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const String& lhs, const char* rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const char* lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

// Overload >= operator
inline bool operator>=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }

inline bool operator>=(const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const String& lhs, const char* rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const char* lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }

// Overload == operator
inline bool operator==(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

inline bool operator==(const String& lhs, const String& rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const String& lhs, const char* rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const char* lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

// Overload != operator
inline bool operator!=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const String& lhs, const char* rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const char* lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline std::ostream& operator<<(std::ostream& out, const String& input) {
  out.write(input.data(), input.size());
  return out;
}

inline int Bytes::memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count) {
  if (lhs == rhs && lhs_count == rhs_count) return 0;

  for (size_t i = 0; i < lhs_count && i < rhs_count; ++i) {
    if (lhs[i] < rhs[i]) return -1;
    if (lhs[i] > rhs[i]) return 1;
  }
  if (lhs_count < rhs_count) {
    return -1;
  } else if (lhs_count > rhs_count) {
    return 1;
  } else {
    return 0;
  }
}
}  // namespace ffi

// Expose to the tvm namespace for usability
// Rationale: no ambiguity even in root
using ffi::Bytes;
using ffi::String;
}  // namespace tvm

namespace std {

template <>
struct hash<::tvm::ffi::Bytes> {
  std::size_t operator()(const ::tvm::ffi::Bytes& bytes) const {
    return ::tvm::ffi::details::StableHashBytes(bytes.data(), bytes.size());
  }
};

template <>
struct hash<::tvm::ffi::String> {
  std::size_t operator()(const ::tvm::ffi::String& str) const {
    return ::tvm::ffi::details::StableHashBytes(str.data(), str.size());
  }
};
}  // namespace std
#endif  // TVM_FFI_STRING_H_
