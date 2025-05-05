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

/*
 * \file tvm/ffi/error.h
 * \brief Error handling component.
 */
#ifndef TVM_FFI_ERROR_H_
#define TVM_FFI_ERROR_H_

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/internal_utils.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

/*!
 * \brief Macro defines whether we enable libbacktrace
 */
#ifndef TVM_FFI_USE_LIBBACKTRACE
#define TVM_FFI_USE_LIBBACKTRACE 1
#endif

/*!
 * \brief Macro defines whether to install signal handler
 *   and print backtrace during segfault
 */
#ifndef TVM_FFI_BACKTRACE_ON_SEGFAULT
#define TVM_FFI_BACKTRACE_ON_SEGFAULT 1
#endif

namespace tvm {
namespace ffi {

/*!
 * \brief Error object class.
 */
class ErrorObj: public Object {
 public:
  /*! \brief The error kind */
  std::string kind;
  /*! \brief Message the error message. */
  std::string message;
  /*! \brief Backtrace, follows python convention(most recent last). */
  std::string backtrace;
  /*! \brief Full message in what_str */
  std::string what_str;

  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIError;
  static constexpr const char* _type_key = "object.Error";

  TVM_FFI_DECLARE_STATIC_OBJECT_INFO(ErrorObj, Object);
};

/*!
 * \brief Managed reference to ErrorObj
 * \sa Error Object
 */
class Error :
  public ObjectRef,
  public std::exception  {
 public:
  Error(std::string kind, std::string message, std::string backtrace) {
    std::ostringstream what;
    what << "Traceback (most recent call last):\n" << backtrace << kind << ": " << message << '\n';
    ObjectPtr<ErrorObj> n = make_object<ErrorObj>();
    n->kind = std::move(kind);
    n->message = std::move(message);
    n->backtrace = std::move(backtrace);
    n->what_str = what.str();
    data_ = std::move(n);
  }

  const char* what() const noexcept(true) override {
    return get()->what_str.c_str();
  }

  TVM_FFI_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Error, ObjectRef, ErrorObj)
};

namespace details {

class ErrorBuilder {
 public:
  explicit ErrorBuilder(std::string kind, std::string traceback, bool log_before_throw)
      : kind_(kind), traceback_(traceback), log_before_throw_(log_before_throw) {}

// MSVC disable warning in error builder as it is exepected
#ifdef _MSC_VER
#pragma disagnostic push
#pragma warning(disable : 4722)
#endif
  // avoid inline to reduce binary size, error throw path do not need to be fast
  [[noreturn]] ~ErrorBuilder() noexcept(false) {
    ::tvm::ffi::Error error(std::move(kind_), stream_.str(), std::move(traceback_));
    if (log_before_throw_) {
      std::cerr << error.what();
    }
    throw error;
  }
#ifdef _MSC_VER
#pragma disagnostic pop
#endif

  std::ostringstream& stream() { return stream_; }

 protected:
  std::string kind_;
  std::ostringstream stream_;
  std::string traceback_;
  bool log_before_throw_;
};

// Code section that depends on dynamic components that requires linking
#if TVM_FFI_ALLOW_DYN_TYPE
/*!
 * \brief Get stack traceback in a string.
 * \param filaname The current file name.
 * \param func The current function
 * \param lineno The current line number
 * \return The traceback string
 *
 * \note filename func and lino are only used as a backup info, most cases they are not needed.
 *  The return value is set to const char* to be more compatible across dll boundaries.
 */
TVM_FFI_DLL const char* Traceback(const char* filename, const char* func, int lineno);
// define traceback here as call into traceback function
#define TVM_FFI_TRACEBACK_HERE ::tvm::ffi::details::Traceback(__FILE__, TVM_FFI_FUNC_SIG, __LINE__)

#else
// simple traceback when allow dyn type is set to false
inline std::string SimpleTraceback(const char* filename, const char* func, int lineno) {
  std::ostringstream traceback;
  // python style backtrace
  traceback << "  " << filename << ", line " << lineno << ", in " << func << '\n';
  return traceback.str();
}
#define TVM_FFI_TRACEBACK_HERE \
  ::tvm::ffi::details::SimpleTraceback(__FILE__, TVM_FFI_FUNC_SIG, __LINE__)
#endif  // TVM_FFI_ALLOW_DYN_TYPE
}  // namespace details

/*!
 * \brief Helper macro to throw an error with traceback and message
 *
 * \code
 *
 *   void ThrowError() {
 *     TVM_FFI_THROW(RuntimeError) << "error message";
 *   }
 *
 * \endcode
 */
#define TVM_FFI_THROW(ErrorKind) \
  ::tvm::ffi::details::ErrorBuilder(#ErrorKind, TVM_FFI_TRACEBACK_HERE, false).stream()

/*!
 * \brief Explicitly log error in stderr and then throw the error.
 *
 * \note This is only necessary on startup functions where we know error
 *  cannot be caught, and it is better to have a clear log message.
 *  In most cases, we should use use TVM_FFI_THROW.
 */
#define TVM_FFI_LOG_AND_THROW(ErrorKind) \
  ::tvm::ffi::details::ErrorBuilder(#ErrorKind, TVM_FFI_TRACEBACK_HERE, true).stream()

// Glog style checks with TVM_FFI prefix
// NOTE: we explicitly avoid glog style generic macros (LOG/CHECK) in tvm ffi
// to avoid potential conflict of downstream users who might have their own GLOG style macros
namespace details {

template <typename X, typename Y>
TVM_FFI_INLINE std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
  std::ostringstream os;
  os << " (" << x << " vs. " << y << ") ";  // CHECK_XX(x, y) requires x and y can be serialized to
                                            // string. Use CHECK(x OP y) otherwise.
  return std::make_unique<std::string>(os.str());
}

#define TVM_FFI_CHECK_FUNC(name, op)                                                   \
  template <typename X, typename Y>                                                    \
  TVM_FFI_INLINE std::unique_ptr<std::string> LogCheck##name(const X& x, const Y& y) { \
    if (x op y) return nullptr;                                                        \
    return LogCheckFormat(x, y);                                                       \
  }                                                                                    \
  TVM_FFI_INLINE std::unique_ptr<std::string> LogCheck##name(int x, int y) {           \
    return LogCheck##name<int, int>(x, y);                                             \
  }

// Inline _Pragma in macros does not work reliably on old version of MSVC and
// GCC. We wrap all comparisons in a function so that we can use #pragma to
// silence bad comparison warnings.
#if defined(__GNUC__) || defined(__clang__)  // GCC and Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#elif defined(_MSC_VER)  // MSVC
#pragma warning(push)
#pragma warning(disable : 4389)  // '==' : signed/unsigned mismatch
#endif

TVM_FFI_CHECK_FUNC(_LT, <)
TVM_FFI_CHECK_FUNC(_GT, >)
TVM_FFI_CHECK_FUNC(_LE, <=)
TVM_FFI_CHECK_FUNC(_GE, >=)
TVM_FFI_CHECK_FUNC(_EQ, ==)
TVM_FFI_CHECK_FUNC(_NE, !=)

#if defined(__GNUC__) || defined(__clang__)  // GCC and Clang
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)  // MSVC
#pragma warning(pop)
#endif
}  // namespace details

#define TVM_FFI_ICHECK_BINARY_OP(name, op, x, y)                        \
  if (auto __tvm__log__err = ::tvm::ffi::details::LogCheck##name(x, y)) \
  TVM_FFI_THROW(InternalError) << "Check failed: " << #x " " #op " " #y << *__tvm__log__err << ": "

#define TVM_FFI_ICHECK(x) \
  if (!(x)) TVM_FFI_THROW(InternalError) << "Check failed: (" #x << ") is false: "

#define TVM_FFI_ICHECK_LT(x, y) TVM_FFI_ICHECK_BINARY_OP(_LT, <, x, y)
#define TVM_FFI_ICHECK_GT(x, y) TVM_FFI_ICHECK_BINARY_OP(_GT, >, x, y)
#define TVM_FFI_ICHECK_LE(x, y) TVM_FFI_ICHECK_BINARY_OP(_LE, <=, x, y)
#define TVM_FFI_ICHECK_GE(x, y) TVM_FFI_ICHECK_BINARY_OP(_GE, >=, x, y)
#define TVM_FFI_ICHECK_EQ(x, y) TVM_FFI_ICHECK_BINARY_OP(_EQ, ==, x, y)
#define TVM_FFI_ICHECK_NE(x, y) TVM_FFI_ICHECK_BINARY_OP(_NE, !=, x, y)
#define TVM_FFI_ICHECK_NOTNULL(x)                                                 \
  ((x) == nullptr ? TVM_FFI_THROW(InternalError) << "Check not null: " #x << ' ', \
   (x)            : (x))  // NOLINT(*)
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_ERROR_H_
