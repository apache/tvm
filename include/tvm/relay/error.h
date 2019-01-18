/*!
 *  Copyright (c) 2018 by Contributors
 * \file error.h
 * \brief The set of errors raised by Relay.
 */
#ifndef TVM_RELAY_ERROR_H_
#define TVM_RELAY_ERROR_H_

#include <string>
#include <vector>
#include <sstream>
#include "./base.h"

namespace tvm {
namespace relay {

/*! \brief A wrapper around std::stringstream.
 *
 * This is designed to avoid platform specific
 * issues compiling and using std::stringstream
 * for error reporting.
 */
struct RelayErrorStream {
  std::stringstream ss;

  template<typename T>
  RelayErrorStream& operator<<(const T& t) {
    ss << t;
    return *this;
  }

  std::string str() const {
    return ss.str();
  }
};

#define RELAY_ERROR(msg) (RelayErrorStream() << msg)

struct Error : public dmlc::Error {
  Span sp;
  explicit Error(const std::string& msg) : dmlc::Error(msg), sp() {}
  Error(const std::stringstream& msg) : dmlc::Error(msg.str()), sp() {} // NOLINT(*)
  Error(const RelayErrorStream& msg) : dmlc::Error(msg.str()), sp() {} // NOLINT(*)
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
