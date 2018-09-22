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

struct Error : public dmlc::Error {
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

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ERROR_H_
