/*!
 *  Copyright (c) 2018 by Contributors
 * \file error.h
 * \brief The set of errors raised by Relay.
 */
#ifndef TVM_RELAY_ERROR_H_
#define TVM_RELAY_ERROR_H_

#include <string>
#include "./base.h"

namespace tvm {
namespace relay {

struct Error : dmlc::Error {
  explicit Error(const std::string &msg) : dmlc::Error(msg) {}
};

struct InternalError : Error {
  explicit InternalError(const std::string &msg) : Error(msg) {}
};

// TODO(@jroesch): we should change spanned errors to report
// errors against the Environment, inverting control to error definition.
struct FatalTypeError : dmlc::Error {
  explicit FatalTypeError(const std::string &s) : dmlc::Error(s) {}
};

struct TypecheckerError : public dmlc::Error {
  explicit TypecheckerError(const std::string &msg) : Error(msg) {}
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ERROR_H_
