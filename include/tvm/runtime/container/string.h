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
 * \file tvm/runtime/container/string.h
 * \brief Runtime String container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_STRING_H_
#define TVM_RUNTIME_CONTAINER_STRING_H_

#include <dmlc/logging.h>
#include <tvm/runtime/container/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
// We use c++14 std::experimental::string_view for optimizing hash computation
// only right now, its usage is limited in this file. Any broader usage of
// std::experiment in our core codebase is discouraged and needs community
// discussion for each use case. Reference for feature test macros of
// string_view:
// https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
#if defined(__cpp_lib_experimental_string_view) && __cpp_lib_experimental_string_view >= 201411
#define TVM_USE_CXX14_STRING_VIEW_HASH 1
#else
#define TVM_USE_CXX14_STRING_VIEW_HASH 0
#endif

// Tested with clang version 9.0.1 and c++17. It will detect string_view support
// correctly.
#if defined(__cpp_lib_string_view) && __cpp_lib_string_view >= 201606
#define TVM_USE_CXX17_STRING_VIEW_HASH 1
#else
#define TVM_USE_CXX17_STRING_VIEW_HASH 0
#endif

#if TVM_USE_CXX17_STRING_VIEW_HASH
#include <string_view>
#elif TVM_USE_CXX14_STRING_VIEW_HASH
#include <experimental/string_view>
#endif

#include <type_traits>
#include <utility>
#include <vector>

namespace llvm {
// String to llvm object compatibility.
class StringRef;
}  // namespace llvm

namespace tvm {
namespace runtime {

// Forward declare TVMArgValue
class TVMArgValue;

/*! \brief An object representing string. It's POD type. */
class StringObj : public Object {
 public:
  /*! \brief The pointer to string data. */
  const char* data;

  /*! \brief The length of the string object. */
  uint64_t size;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeString;
  static constexpr const char* _type_key = "runtime.String";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringObj, Object);

 private:
  /*! \brief String object which is moved from std::string container. */
  class FromStd;

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
  /*!
   * \brief Construct an empty string.
   */
  String() : String(std::string()) {}
  /*!
   * \brief Construct a new String object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  String(std::string other);  // NOLINT(*)

  /*!
   * \brief Construct a new String object
   *
   * \param other a char array.
   */
  String(const char* other)  // NOLINT(*)
      : String(std::string(other)) {}

  /*!
   * \brief Construct a new null object
   */
  String(std::nullptr_t)  // NOLINT(*)
      : ObjectRef(nullptr) {}

  /*!
   * \brief Change the value the reference object points to.
   *
   * \param other The value for the new String
   *
   */
  inline String& operator=(std::string other);

  /*!
   * \brief Change the value the reference object points to.
   *
   * \param other The value for the new String
   */
  inline String& operator=(const char* other);

  /*!
   * \brief Compares this String object to other
   *
   * \param other The String to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const String& other) const {
    return memncmp(data(), other.data(), size(), other.size());
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
    return memncmp(data(), other.data(), size(), other.size());
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
    return memncmp(data(), other, size(), std::strlen(other));
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

  // LLVM compatibility function, implemented in src/target/llvm/llvm_common.h
  /*!
   * \brief Convert String to an llvm::StringRef object
   *
   * \return llvm::StringRef
   */
  inline operator llvm::StringRef() const;

  /*!
   * \brief Check if a TVMArgValue can be converted to String, i.e. it can be std::string or String
   * \param val The value to be checked
   * \return A boolean indicating if val can be converted to String
   */
  inline static bool CanConvertFrom(const TVMArgValue& val);

  /*!
   * \brief Hash the binary bytes
   * \param data The data pointer
   * \param size The size of the bytes.
   * \return the hash value.
   */
  static size_t HashBytes(const char* data, size_t size) {
    // This function falls back to string copy with c++11 compiler and is
    // recommended to be compiled with c++14
#if TVM_USE_CXX17_STRING_VIEW_HASH
    return std::hash<std::string_view>()(std::string_view(data, size));
#elif TVM_USE_CXX14_STRING_VIEW_HASH
    return std::hash<std::experimental::string_view>()(std::experimental::string_view(data, size));
#else
    return std::hash<std::string>()(std::string(data, size));
#endif
  }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(String, ObjectRef, StringObj);

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

  friend struct tvm::runtime::ObjectEqual;
};

/*! \brief An object representing string moved from std::string. */
class StringObj::FromStd : public StringObj {
 public:
  /*!
   * \brief Construct a new FromStd object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  explicit FromStd(std::string other) : data_container{other} {}

 private:
  /*! \brief Container that holds the memory. */
  std::string data_container;

  friend class String;
};

inline String::String(std::string other) {
  auto ptr = make_object<StringObj::FromStd>(std::move(other));
  ptr->size = ptr->data_container.size();
  ptr->data = ptr->data_container.data();
  data_ = std::move(ptr);
}

inline String& String::operator=(std::string other) {
  String replace{std::move(other)};
  data_.swap(replace.data_);
  return *this;
}

inline String& String::operator=(const char* other) { return operator=(std::string(other)); }

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

inline bool operator>=(const char* lhs, const String& rhs) { return rhs.compare(rhs) <= 0; }

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

inline int String::memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count) {
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

inline size_t ObjectHash::operator()(const ObjectRef& a) const {
  if (const auto* str = a.as<StringObj>()) {
    return String::HashBytes(str->data, str->size);
  }
  return ObjectPtrHash()(a);
}

inline bool ObjectEqual::operator()(const ObjectRef& a, const ObjectRef& b) const {
  if (a.same_as(b)) {
    return true;
  }
  if (const auto* str_a = a.as<StringObj>()) {
    if (const auto* str_b = b.as<StringObj>()) {
      return String::memncmp(str_a->data, str_b->data, str_a->size, str_b->size) == 0;
    }
  }
  return false;
}
}  // namespace runtime

// expose the functions to the root namespace.
using runtime::String;
using runtime::StringObj;
}  // namespace tvm

namespace std {

template <>
struct hash<::tvm::runtime::String> {
  std::size_t operator()(const ::tvm::runtime::String& str) const {
    return ::tvm::runtime::String::HashBytes(str.data(), str.size());
  }
};
}  // namespace std

#endif  // TVM_RUNTIME_CONTAINER_STRING_H_
