/*!
 *  Copyright (c) 2018 by Contributors
 * \file error_reporter.h
 * \brief The set of errors raised by Relay.
 */
#ifndef TVM_RELAY_ERROR_REPORTER_H_
#define TVM_RELAY_ERROR_REPORTER_H_

#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <string>
#include <sstream>
#include <vector>

namespace tvm {
namespace relay {

/*! \brief An abstraction around how errors are stored and reported.
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
 * of the error and no indication of where it occured in the original program.
 *
 * The latter mode is not ideal, and the goal of the new error reporting machinery is
 * to avoid ever reporting errors in this style.
 */
class ErrorReporter {
 public:
  ErrorReporter() : errors_(), node_to_error_() {}

  /*! \brief Report a tvm::relay::Error.
   *
   * This API is useful for reporting spanned errors.
   *
   * \param err The error to report.
   */
  void Report(const Error& err) {
    if (!err.sp.defined()) {
      throw err;
    }

    this->errors_.push_back(err);
  }

  /*! \brief Report an error against a program, using the full program
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
  inline void ReportAt(const GlobalVar& global, const NodeRef& node, std::stringstream& err) {
    this->ReportAt(global, node, Error(err));
  }

  /*! \brief Report an error against a program, using the full program
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
  void ReportAt(const GlobalVar& global, const NodeRef& node, const Error& err);

  /*! \brief Render all reported errors and exit the program.
   *
   * This function should be used after executing a pass to render reported errors.
   *
   * It will build an error message from the set of errors, depending on the error
   * reporting strategy.
   *
   * \param module The module to report errors on.
   * \param use_color Controls whether to colorize the output.
   */
  [[noreturn]] void RenderErrors(const Module& module, bool use_color = true);

  inline bool AnyErrors() {
    return errors_.size() != 0;
  }

 private:
  std::vector<Error> errors_;
  std::unordered_map<NodeRef, std::vector<size_t>, NodeHash, NodeEqual> node_to_error_;
  std::unordered_map<NodeRef, GlobalVar, NodeHash, NodeEqual> node_to_gv_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ERROR_REPORTER_H_
