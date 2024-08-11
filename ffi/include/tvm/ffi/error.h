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

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>

#include <string>
#include <sstream>

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
  explicit ErrorBuilder(const char* kind, const char* filename, const char* func, int32_t lineno)
      : kind_(kind) {
    std::ostringstream backtrace;
    // python style backtrace
    backtrace << "  " << filename << ", line " << lineno << ", in " << func << '\n';
    backtrace_ = backtrace.str();
  }

// MSVC disable warning in error builder as it is exepected
#ifdef _MSC_VER
#pragma disagnostic push
#pragma warning(disable : 4722)
#endif
  [[noreturn]] ~ErrorBuilder() noexcept(false) {
    throw ::tvm::ffi::Error(std::move(kind_), message_.str(), std::move(backtrace_));
  }
#ifdef _MSC_VER
#pragma disagnostic pop
#endif

  std::ostringstream &Get() { return message_; }

protected:
  std::string kind_;
  std::ostringstream message_;
  std::string backtrace_;
};
}  // namespace details

/*!
 * \brief Helper macro to throw an error with backtrace and message
 */
#define TVM_FFI_THROW(ErrorKind) \
  ::tvm::ffi::details::ErrorBuilder(#ErrorKind, __FILE__, TVM_FFI_FUNC_SIG, __LINE__).Get()

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_ERROR_H_
