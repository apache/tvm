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
#ifndef TVM_RELAY_ERROR_H_
#define TVM_RELAY_ERROR_H_

#include <tvm/ir/module.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {
/*!
 * \brief A wrapper around std::stringstream to build error.
 *include/tvm/ir/type.h
 * Can be consumed by CompileError to construct an error.
 *
 * \code
 *
 * void ReportError(const CompileError& err);
 *
 * void Test(int number) {
 *   // Use error reporter to construct an error.
 *   ReportError(ErrorBuilder() << "This is an error number=" << number);
 * }
 *
 * \endcode
 */
struct ErrorBuilder {
 public:
  template <typename T>
  ErrorBuilder& operator<<(const T& val) {  // NOLINT(*)
    stream_ << val;
    return *this;
  }

 private:
  std::stringstream stream_;
  friend class CompileError;
};

/*!
 * \brief Custom Error class to be thrown during compilation.
 */
class CompileError : public Error {
 public:
  /*! \brief Location of the error */
  Span span;
  /*!
   * \brief construct error from message.
   * \param msg The message
   */
  explicit CompileError(const std::string& msg) : Error(msg), span(nullptr) {}
  /*!
   * \brief construct error from error builder.
   * \param err The error builder
   */
  CompileError(const ErrorBuilder& err) : Error(err.stream_.str()), span(nullptr) {}  // NOLINT(*)
  /*!
   * \brief copy constructor.
   * \param other The other ereor.
   */
  CompileError(const CompileError& other) : Error(other.what()), span(other.span) {}  // NOLINT(*)
  /*!
   * \brief default constructor. */
  CompileError() : Error(""), span(nullptr) {}
};

/*!
 * \brief An abstraction around how errors are stored and reported.
 * Designed to be opaque to users, so we can support a robust and simpler
 * error reporting mode, as well as a more complex mode.
 *
 * The first mode is the most accurate: we report a Relay error at a specific
 * Span, and then render the error message directly against a textual representation
 * of the program, highlighting the exact lines in which it occurs. This mode is not
 * implemented in this PR and will not work.
 *
 * The second mode is a general-purpose mode, which attempts to annotate the program's
 * textual format with errors.
 *
 * The final mode represents the old mode, if we report an error that has no span or
 * expression, we will default to throwing an exception with a textual representation
 * of the error and no indication of where it occurred in the original program.
 *
 * The latter mode is not ideal, and the goal of the new error reporting machinery is
 * to avoid ever reporting errors in this style.
 */
class ErrorReporter {
 public:
  /*! \brief default constructor. */
  ErrorReporter() : errors_(), node_to_error_() {}

  /*!
   * \brief Report a CompileError.
   *
   * This API is useful for reporting spanned errors.
   *
   * \param err The error to report.
   */
  void Report(const CompileError& err) {
    if (!err.span.defined()) {
      throw err;
    }

    this->errors_.push_back(err);
  }

  /*!
   * \brief Report an error against a program, using the full program
   * error reporting strategy.
   *
   * This error reporting method requires the global function in which
   * to report an error, the expression to report the error on,
   * and the error object.
   *
   * \param global The global function in which the expression is contained.
   * \param node The expression or type to report the error at.
   * \param err The error message to report.
   */
  void ReportAt(const GlobalVar& global, const ObjectRef& node, std::stringstream& err) {
    std::string err_msg = err.str();
    this->ReportAt(global, node, CompileError(err_msg));
  }

  /*!
   * \brief Report an error against a program, using the full program
   * error reporting strategy.
   *
   * This error reporting method requires the global function in which
   * to report an error, the expression to report the error on,
   * and the error object.
   *
   * \param global The global function in which the expression is contained.
   * \param node The expression or type to report the error at.
   * \param err The error to report.
   */
  void ReportAt(const GlobalVar& global, const ObjectRef& node, const CompileError& err);

  /*!
   * \brief Render all reported errors and exit the program.
   *
   * This function should be used after executing a pass to render reported errors.
   *
   * It will build an error message from the set of errors, depending on the error
   * reporting strategy.
   *
   * \param module The module to report errors on.
   * \param use_color Controls whether to colorize the output.
   */
  void RenderErrors(const IRModule& module, bool use_color = true);

  inline bool AnyErrors() { return errors_.size() != 0; }

 private:
  std::vector<CompileError> errors_;
  std::unordered_map<ObjectRef, std::vector<size_t>, ObjectPtrHash, ObjectPtrEqual> node_to_error_;
  std::unordered_map<ObjectRef, GlobalVar, ObjectPtrHash, ObjectPtrEqual> node_to_gv_;
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ERROR_H_
