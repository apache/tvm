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
    Error(std::string msg) : dmlc::Error(msg) {}
};

struct SpannedError {
  std::string msg;
  Span sp;
  SpannedError(std::string msg, Span sp) : msg(msg), sp(sp) {}
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ERROR_H_
