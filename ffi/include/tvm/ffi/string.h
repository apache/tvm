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
namespace details {
/*!
 * \brief Base class for bytes and string objects.
 */
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

/*!
 * \brief Helper cell class that can be used to back small string
 * \note Do not use directly, use String or Bytes instead
 */
class BytesBaseCell {
 public:
  BytesBaseCell() {
    // initialize to none
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }

  explicit BytesBaseCell(std::nullopt_t) {
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }

  BytesBaseCell(const BytesBaseCell& other) : data_(other.data_) {  // NOLINT(*)
    if (data_.type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectUnsafe::IncRefObjectHandle(data_.v_obj);
    }
  }

  BytesBaseCell(BytesBaseCell&& other) : data_(other.data_) {  // NOLINT(*)
    other.data_.type_index = TypeIndex::kTVMFFINone;
  }

  BytesBaseCell& operator=(const BytesBaseCell& other) {
    BytesBaseCell(other).swap(*this);  // NOLINT(*)
    return *this;
  }

  BytesBaseCell& operator=(BytesBaseCell&& other) {
    BytesBaseCell(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }

  ~BytesBaseCell() {
    if (data_.type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectUnsafe::DecRefObjectHandle(data_.v_obj);
    }
  }

  /*!
   * \brief Check if the cell is null
   * \return true if the cell is null, false otherwise
   */
  bool operator==(std::nullopt_t) const { return data_.type_index == TypeIndex::kTVMFFINone; }

  /*!
   * \brief Check if the cell is not null
   * \return true if the cell is not null, false otherwise
   */
  bool operator!=(std::nullopt_t) const { return data_.type_index != TypeIndex::kTVMFFINone; }

  /*!
   * \brief Swap this String with another string
   * \param other The other string
   */
  void swap(BytesBaseCell& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }

  const char* data() const noexcept {
    if (data_.type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      return data_.v_bytes;
    } else {
      return TVMFFIBytesGetByteArrayPtr(data_.v_obj)->data;
    }
  }

  size_t size() const noexcept {
    if (data_.type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      return data_.small_str_len;
    } else {
      return TVMFFIBytesGetByteArrayPtr(data_.v_obj)->size;
    }
  }

  template <typename LargeObj>
  void InitFromStd(std::string&& other, int32_t large_type_index) {
    // needs to be reset to none first for exception safety
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&data_);
    ObjectPtr<LargeObj> ptr = make_object<BytesObjStdImpl<LargeObj>>(std::move(other));
    data_.v_obj = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(ptr));
    data_.type_index = large_type_index;
  }

  /*!
   * \brief Create a new empty space for a string
   * \param size The size of the string
   * \param small_type_index The type index for the small string
   * \param large_type_index The type index for the large string
   * \note always reserve one byte for \0 compactibility
   * \return A pointer to the empty space
   */
  template <typename LargeObj>
  char* InitSpaceForSize(size_t size, int32_t small_type_index, int32_t large_type_index) {
    size_t kMaxSmallBytesLen = sizeof(int64_t) - 1;
    // first zero the content, this is important for exception safety
    data_.type_index = small_type_index;
    data_.zero_padding = 0;
    if (size <= kMaxSmallBytesLen) {
      // set up the size accordingly
      data_.small_str_len = static_cast<uint32_t>(size);
      return data_.v_bytes;
    } else {
      // allocate from heap
      ObjectPtr<LargeObj> ptr = make_inplace_array_object<LargeObj, char>(size + 1);
      char* dest_data = reinterpret_cast<char*>(ptr.get()) + sizeof(LargeObj);
      ptr->data = dest_data;
      ptr->size = size;
      TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&data_);
      data_.v_obj = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(ptr));
      // now reset the type index to str
      data_.type_index = large_type_index;
      return dest_data;
    }
  }

  void InitTypeIndex(int32_t type_index) { data_.type_index = type_index; }

  void MoveToAny(TVMFFIAny* result) {
    *result = data_;
    data_.type_index = TypeIndex::kTVMFFINone;
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }

  TVMFFIAny CopyToTVMFFIAny() const { return data_; }

  static BytesBaseCell CopyFromAnyView(const TVMFFIAny* src) {
    BytesBaseCell result(*src);
    if (result.data_.type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectUnsafe::IncRefObjectHandle(result.data_.v_obj);
    }
    return result;
  }

  static BytesBaseCell MoveFromAny(TVMFFIAny* src) {
    BytesBaseCell result(*src);
    src->type_index = TypeIndex::kTVMFFINone;
    src->zero_padding = 0;
    src->v_int64 = 0;
    return result;
  }

 private:
  explicit BytesBaseCell(TVMFFIAny data) : data_(data) {}
  /*! \brief internal backing data */
  TVMFFIAny data_;
};
}  // namespace details

/*!
 * \brief Managed reference of byte array.
 */
class Bytes {
 public:
  /*! \brief default constructor */
  Bytes() { data_.InitTypeIndex(TypeIndex::kTVMFFISmallBytes); }
  /*!
   * \brief constructor from size
   *
   * \param other a char array.
   */
  Bytes(const char* data, size_t size) { this->InitData(data, size); }
  /*!
   * \brief constructor from TVMFFIByteArray
   *
   * \param other a char array.
   */
  Bytes(TVMFFIByteArray bytes) {  // NOLINT(*)
    this->InitData(bytes.data, bytes.size);
  }
  /*!
   * \brief constructor from std::string
   *
   * \param other a char array.
   */
  Bytes(const std::string& other) {  // NOLINT(*)
    this->InitData(other.data(), other.size());
  }
  /*!
   * \brief constructor from std::string
   *
   * \param other a char array.
   */
  Bytes(std::string&& other) {  // NOLINT(*)
    data_.InitFromStd<details::BytesObj>(std::move(other), TypeIndex::kTVMFFIBytes);
  }
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
  size_t size() const { return data_.size(); }
  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  const char* data() const { return data_.data(); }
  /*!
   * \brief Convert String to an std::string object
   *
   * \return std::string
   */
  operator std::string() const { return std::string{data(), size()}; }

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
  static int memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count) {
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
  /*!
   * \brief Compare two char sequence for equality
   *
   * \param lhs Pointers to the char array to compare
   * \param rhs Pointers to the char array to compare
   * \param lhs_count Length of the char array to compare
   * \param rhs_count Length of the char array to compare
   *
   * \return true if the two char sequences are equal, false otherwise.
   */
  static bool memequal(const void* lhs, const void* rhs, size_t lhs_count, size_t rhs_count) {
    return lhs_count == rhs_count && (lhs == rhs || std::memcmp(lhs, rhs, lhs_count) == 0);
  }

 private:
  template <typename, typename>
  friend struct TypeTraits;
  template <typename, typename>
  friend class Optional;
  // internal backing cell
  details::BytesBaseCell data_;
  // create a new String from TVMFFIAny, must keep private
  explicit Bytes(details::BytesBaseCell data) : data_(data) {}
  char* InitSpaceForSize(size_t size) {
    return data_.InitSpaceForSize<details::BytesObj>(size, TypeIndex::kTVMFFISmallBytes,
                                                     TypeIndex::kTVMFFIBytes);
  }
  void InitData(const char* data, size_t size) {
    char* dest_data = InitSpaceForSize(size);
    std::memcpy(dest_data, data, size);
    // mainly to be compat with string
    dest_data[size] = '\0';
  }
};

/*!
 * \brief String container class.
 */
class String {
 public:
  /*!
   * \brief avoid misuse of nullptr
   */
  String(std::nullptr_t) = delete;  // NOLINT(*)
  /*!
   * \brief constructor
   */
  String() { data_.InitTypeIndex(TypeIndex::kTVMFFISmallStr); }
  // constructors from Any
  String(const String& other) = default;             // NOLINT(*)
  String(String&& other) = default;                  // NOLINT(*)
  String& operator=(const String& other) = default;  // NOLINT(*)
  String& operator=(String&& other) = default;       // NOLINT(*)

  /*!
   * \brief Swap this String with another string
   * \param other The other string
   */
  void swap(String& other) noexcept {  // NOLINT(*)
    std::swap(data_, other.data_);
  }

  String& operator=(const std::string& other) {
    String(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  String& operator=(std::string&& other) {
    String(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }

  String& operator=(const char* other) {
    String(other).swap(*this);  // NOLINT(*)
    return *this;
  }

  /*!
   * \brief constructor from raw string
   *
   * \param other a char array.
   */
  String(const char* other, size_t size) { this->InitData(other, size); }

  /*!
   * \brief constructor from raw string
   *
   * \param other a char array.
   * \note This constructor is marked as explicit to avoid implicit conversion
   *       of nullptr value here to string, which then was used in comparison
   */
  String(const char* other) {  // NOLINT(*)
    this->InitData(other, std::char_traits<char>::length(other));
  }
  /*!
   * \brief Construct a new string object
   * \param other The std::string object to be copied
   */
  String(const std::string& other) {  // NOLINT(*)
    this->InitData(other.data(), other.size());
  }

  /*!
   * \brief Construct a new string object
   * \param other The std::string object to be moved
   */
  String(std::string&& other) {  // NOLINT(*)
    // exception safety, first set to none so if exception is thrown
    // destructor works correctly
    data_.InitFromStd<details::StringObj>(std::move(other), TypeIndex::kTVMFFIStr);
  }

  /*!
   * \brief constructor from TVMFFIByteArray
   *
   * \param other a TVMFFIByteArray.
   */
  explicit String(TVMFFIByteArray other) { this->InitData(other.data, other.size); }

  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  const char* data() const noexcept { return data_.data(); }

  /*!
   * \brief Returns a pointer to the char array in the string.
   *
   * \return const char*
   */
  const char* c_str() const noexcept { return data(); }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t size() const noexcept { return data_.size(); }

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
    const char* this_data = data();
    size_t this_size = size();
    for (size_t i = 0; i < this_size; ++i) {
      // other is shorter than this
      if (other[i] == '\0') return 1;
      if (this_data[i] < other[i]) return -1;
      if (this_data[i] > other[i]) return 1;
    }
    // other equals this
    if (other[this_size] == '\0') return 0;
    // other longer than this
    return -1;
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
   * \brief Convert String to an std::string object
   *
   * \return std::string
   */
  operator std::string() const { return std::string{data(), size()}; }

 private:
  template <typename, typename>
  friend struct TypeTraits;
  template <typename, typename>
  friend class Optional;
  // internal backing cell
  details::BytesBaseCell data_;
  // create a new String from TVMFFIAny, must keep private
  explicit String(details::BytesBaseCell data) : data_(data) {}
  /*!
   * \brief Create a new empty space for a string
   * \param size The size of the string
   * \return A pointer to the empty space
   */
  char* InitSpaceForSize(size_t size) {
    return data_.InitSpaceForSize<details::StringObj>(size, TypeIndex::kTVMFFISmallStr,
                                                      TypeIndex::kTVMFFIStr);
  }
  void InitData(const char* data, size_t size) {
    char* dest_data = InitSpaceForSize(size);
    std::memcpy(dest_data, data, size);
    dest_data[size] = '\0';
  }
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
    String ret;
    // disable stringop-overflow and restrict warnings
    // gcc may produce false positive when we enable dest_data returned from small string path
    // Because compiler is not able to detect the condition that the path is only triggered via
    // size < kMaxSmallStrLen and can report it as a overflow case.
#if (__GNUC__) && !(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wrestrict"
#endif
    char* dest_data = ret.InitSpaceForSize(lhs_size + rhs_size);
    std::memcpy(dest_data, lhs, lhs_size);
    std::memcpy(dest_data + lhs_size, rhs, rhs_size);
    dest_data[lhs_size + rhs_size] = '\0';
#if (__GNUC__) && !(__clang__)
#pragma GCC diagnostic pop
#endif
    return ret;
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

template <>
inline constexpr bool use_default_type_traits_v<Bytes> = false;

// specialize to enable implicit conversion from TVMFFIByteArray*
template <>
struct TypeTraits<Bytes> : public TypeTraitsBase {
  // bytes can be union type of small bytes and object, so keep it as any
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;

  TVM_FFI_INLINE static void CopyToAnyView(const Bytes& src, TVMFFIAny* result) {
    *result = src.data_.CopyToTVMFFIAny();
  }

  TVM_FFI_INLINE static void MoveToAny(Bytes src, TVMFFIAny* result) {
    src.data_.MoveToAny(result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFISmallBytes ||
           src->type_index == TypeIndex::kTVMFFIBytes;
  }

  TVM_FFI_INLINE static Bytes CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return Bytes(details::BytesBaseCell::CopyFromAnyView(src));
  }

  TVM_FFI_INLINE static Bytes MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return Bytes(details::BytesBaseCell::MoveFromAny(src));
  }

  TVM_FFI_INLINE static std::optional<Bytes> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIByteArrayPtr) {
      return Bytes(*static_cast<TVMFFIByteArray*>(src->v_ptr));
    }
    if (src->type_index == TypeIndex::kTVMFFISmallBytes ||
        src->type_index == TypeIndex::kTVMFFIBytes) {
      return Bytes(details::BytesBaseCell::CopyFromAnyView(src));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "bytes"; }
};

template <>
inline constexpr bool use_default_type_traits_v<String> = false;

// specialize to enable implicit conversion from const char*
template <>
struct TypeTraits<String> : public TypeTraitsBase {
  // string can be union type of small string and object, so keep it as any
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;

  TVM_FFI_INLINE static void CopyToAnyView(const String& src, TVMFFIAny* result) {
    *result = src.data_.CopyToTVMFFIAny();
  }

  TVM_FFI_INLINE static void MoveToAny(String src, TVMFFIAny* result) {
    src.data_.MoveToAny(result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFISmallStr ||
           src->type_index == TypeIndex::kTVMFFIStr;
  }

  TVM_FFI_INLINE static String CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return String(details::BytesBaseCell::CopyFromAnyView(src));
  }

  TVM_FFI_INLINE static String MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return String(details::BytesBaseCell::MoveFromAny(src));
  }

  TVM_FFI_INLINE static std::optional<String> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIRawStr) {
      return String(src->v_c_str);
    }
    if (src->type_index == TypeIndex::kTVMFFISmallStr || src->type_index == TypeIndex::kTVMFFIStr) {
      return String(details::BytesBaseCell::CopyFromAnyView(src));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "str"; }
};

// const char*, requirement: not nullable, do not retain ownership
template <int N>
struct TypeTraits<char[N]> : public TypeTraitsBase {
  // NOTE: only enable implicit conversion into AnyView
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static void CopyToAnyView(const char src[N], TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIRawStr;
    result->zero_padding = 0;
    result->v_c_str = src;
  }

  TVM_FFI_INLINE static void MoveToAny(const char src[N], TVMFFIAny* result) {
    // when we need to move to any, convert to owned object first
    TypeTraits<String>::MoveToAny(String(src), result);
  }
};

template <>
struct TypeTraits<const char*> : public TypeTraitsBase {
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static void CopyToAnyView(const char* src, TVMFFIAny* result) {
    TVM_FFI_ICHECK_NOTNULL(src);
    result->type_index = TypeIndex::kTVMFFIRawStr;
    result->zero_padding = 0;
    result->v_c_str = src;
  }

  TVM_FFI_INLINE static void MoveToAny(const char* src, TVMFFIAny* result) {
    // when we need to move to any, convert to owned object first
    TypeTraits<String>::MoveToAny(String(src), result);
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
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_ptr = src;
  }

  TVM_FFI_INLINE static void MoveToAny(TVMFFIByteArray* src, TVMFFIAny* result) {
    TypeTraits<Bytes>::MoveToAny(Bytes(*src), result);
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
inline constexpr bool use_default_type_traits_v<std::string> = false;

template <>
struct TypeTraits<std::string>
    : public FallbackOnlyTraitsBase<std::string, const char*, TVMFFIByteArray*, Bytes, String> {
  TVM_FFI_INLINE static void CopyToAnyView(const std::string& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIRawStr;
    result->zero_padding = 0;
    result->v_c_str = src.c_str();
  }

  TVM_FFI_INLINE static void MoveToAny(std::string src, TVMFFIAny* result) {
    // when we need to move to any, convert to owned object first
    TypeTraits<String>::MoveToAny(String(std::move(src)), result);
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
inline bool operator<(std::nullptr_t, const String& rhs) = delete;
inline bool operator<(const String& lhs, std::nullptr_t) = delete;

inline bool operator<(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

inline bool operator<(const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const String& lhs, const char* rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const char* lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

// Overload > operator
inline bool operator>(std::nullptr_t, const String& rhs) = delete;
inline bool operator>(const String& lhs, std::nullptr_t) = delete;

inline bool operator>(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

inline bool operator>(const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const String& lhs, const char* rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const char* lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

// Overload <= operator
inline bool operator<=(std::nullptr_t, const String& rhs) = delete;
inline bool operator<=(const String& lhs, std::nullptr_t) = delete;

inline bool operator<=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

inline bool operator<=(const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const String& lhs, const char* rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const char* lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

// Overload >= operator
inline bool operator>=(std::nullptr_t, const String& rhs) = delete;
inline bool operator>=(const String& lhs, std::nullptr_t) = delete;

inline bool operator>=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }

inline bool operator>=(const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const String& lhs, const char* rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const char* lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }

// delete Overload == operator for nullptr
inline bool operator==(const String& lhs, std::nullptr_t) = delete;
inline bool operator==(std::nullptr_t, const String& rhs) = delete;

inline bool operator==(const String& lhs, const std::string& rhs) {
  return Bytes::memequal(lhs.data(), rhs.data(), lhs.size(), rhs.size());
}

inline bool operator==(const std::string& lhs, const String& rhs) {
  return Bytes::memequal(lhs.data(), rhs.data(), lhs.size(), rhs.size());
}

inline bool operator==(const String& lhs, const String& rhs) {
  return Bytes::memequal(lhs.data(), rhs.data(), lhs.size(), rhs.size());
}

inline bool operator==(const String& lhs, const char* rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const char* lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

// Overload != operator
inline bool operator!=(const String& lhs, std::nullptr_t) = delete;
inline bool operator!=(std::nullptr_t, const String& rhs) = delete;

inline bool operator!=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const String& lhs, const char* rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const char* lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline std::ostream& operator<<(std::ostream& out, const String& input) {
  out.write(input.data(), input.size());
  return out;
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
    return std::hash<std::string_view>()(std::string_view(bytes.data(), bytes.size()));
  }
};

template <>
struct hash<::tvm::ffi::String> {
  std::size_t operator()(const ::tvm::ffi::String& str) const {
    return std::hash<std::string_view>()(std::string_view(str.data(), str.size()));
  }
};
}  // namespace std
#endif  // TVM_FFI_STRING_H_
