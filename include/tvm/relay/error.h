/*!
 *  Copyright (c) 2018 by Contributors
 * \file error.h
 * \brief The set of errors raised by Relay.
 */
#ifndef TVM_RELAY_ERROR_H_
#define TVM_RELAY_ERROR_H_

#include <string>
#include "./base.h"
#include "./source_map.h"

namespace tvm {
namespace relay {

struct Error : public dmlc::Error {
  Span sp;
  explicit Error(const std::string &msg) : dmlc::Error(msg) {}
};

struct InternalError : public Error {
  explicit InternalError(const std::string &msg) : Error(msg) {}
};

struct FatalTypeError : public Error {
  explicit FatalTypeError(const std::string &s) : Error(s) {}
};

struct TypecheckerError : public Error {
  explicit TypecheckerError(const std::string &msg) : Error(msg) {}
};

class ErrorReporter {
public:
  SourceMap src_map;
  std::vector<Error> errors;

  ErrorReporter() : src_map(), errors() {}
  ErrorReporter(SourceMap src_map) : errors() {}

  void Report(const Error& err) {
    this->errors.push_back(err);
  }

  dmlc::Error Render();
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ERROR_H_
