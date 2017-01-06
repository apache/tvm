/*!
 * Copyright (c) 2016 by Contributors
 * \file optional.h
 * \brief Container to hold optional data.
 */
#ifndef DMLC_OPTIONAL_H_
#define DMLC_OPTIONAL_H_

#include <iostream>
#include <utility>
#include <algorithm>

#include "./base.h"
#include "./logging.h"
#include "./type_traits.h"

namespace dmlc {

/*! \brief dummy type for assign null to optional */
struct nullopt_t {
#if defined(_MSC_VER) && _MSC_VER < 1900
  /*! \brief dummy constructor */
  explicit nullopt_t(int) {}
#else
  /*! \brief dummy constructor */
  constexpr nullopt_t(int) {}
#endif
};

/*! Assign null to optional: optional<T> x = nullopt; */
constexpr const nullopt_t nullopt = nullopt_t(0);

/*!
 * \brief c++17 compatible optional class.
 *
 * At any time an optional<T> instance either
 * hold no value (string representation "None")
 * or hold a value of type T.
 */
template<typename T>
class optional {
 public:
  /*! \brief construct an optional object that contains no value */
  optional() : is_none(true) {}
  /*! \brief construct an optional object with value */
  explicit optional(const T& value) {
    is_none = false;
    new (&val) T(value);
  }
  /*! \brief construct an optional object with another optional object */
  optional(const optional<T>& other) {
    is_none = other.is_none;
    if (!is_none) {
      new (&val) T(other.value());
    }
  }
  /*! \brief deconstructor */
  ~optional() {
    if (!is_none) {
      reinterpret_cast<T*>(&val)->~T();
    }
  }
  /*! \brief swap two optional */
  void swap(optional<T>& other) {
    std::swap(val, other.val);
    std::swap(is_none, other.is_none);
  }
  /*! \brief set this object to hold value
   *  \param value the value to hold
   *  \return return self to support chain assignment
   */
  optional<T>& operator=(const T& value) {
    (optional<T>(value)).swap(*this);
    return *this;
  }
  /*! \brief set this object to hold the same value with other
   *  \param other the other object
   *  \return return self to support chain assignment
   */
  optional<T>& operator=(const optional<T> &other) {
    (optional<T>(other)).swap(*this);
    return *this;
  }
  /*! \brief clear the value this object is holding.
   *         optional<T> x = nullopt;
   */
  optional<T>& operator=(nullopt_t) {
    (optional<T>()).swap(*this);
    return *this;
  }
  /*! \brief non-const dereference operator */
  T& operator*() {  // NOLINT(*)
    return *reinterpret_cast<T*>(&val);
  }
  /*! \brief const dereference operator */
  const T& operator*() const {
    return *reinterpret_cast<const T*>(&val);
  }
  /*! \brief return the holded value.
   *         throws std::logic_error if holding no value
   */
  const T& value() const {
    if (is_none) {
      throw std::logic_error("bad optional access");
    }
    return *reinterpret_cast<const T*>(&val);
  }
  /*! \brief whether this object is holding a value */
  explicit operator bool() const { return !is_none; }

 private:
  // whether this is none
  bool is_none;
  // on stack storage of value
  typename std::aligned_storage<sizeof(T), alignof(T)>::type val;
};

/*! \brief serialize an optional object to string.
 *
 *  \code
 *    dmlc::optional<int> x;
 *    std::cout << x;  // None
 *    x = 0;
 *    std::cout << x;  // 0
 *  \endcode
 *
 *  \param os output stream
 *  \param t source optional<T> object
 *  \return output stream
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, const optional<T> &t) {
  if (t) {
    os << *t;
  } else {
    os << "None";
  }
  return os;
}

/*! \brief parse a string object into optional<T>
 *
 *  \code
 *    dmlc::optional<int> x;
 *    std::string s1 = "1";
 *    std::istringstream is1(s1);
 *    s1 >> x;  // x == optional<int>(1)
 *
 *    std::string s2 = "None";
 *    std::istringstream is2(s2);
 *    s2 >> x;  // x == optional<int>()
 *  \endcode
 *
 *  \param is input stream
 *  \param t target optional<T> object
 *  \return input stream
 */
template<typename T>
std::istream &operator>>(std::istream &is, optional<T> &t) {
  char buf[4];
  std::streampos origin = is.tellg();
  is.read(buf, 4);
  if (is.fail() || buf[0] != 'N' || buf[1] != 'o' ||
      buf[2] != 'n' || buf[3] != 'e') {
    is.clear();
    is.seekg(origin);
    T x;
    is >> x;
    t = x;
  } else {
    t = nullopt;
  }
  return is;
}

/*! \brief description for optional int */
DMLC_DECLARE_TYPE_NAME(optional<int>, "int or None");

}  // namespace dmlc

#endif  // DMLC_OPTIONAL_H_
