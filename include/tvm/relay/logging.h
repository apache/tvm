/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/logging.h
 * \brief A wrapper around dmlc-core/logging.h which adds the ability
 * to toggle logging via an environment variable.
 */

#ifndef TVM_RELAY_LOGGING_H_
#define TVM_RELAY_LOGGING_H_

#include <dmlc/logging.h>
#include <string>
#include <cstdlib>
#include <iostream>

namespace tvm {
namespace relay {

static bool logging_enabled() {
  if (auto var = std::getenv("RELAY_LOG")) {
    std::string is_on(var);
    return is_on == "1";
  } else {
      return false;
  }
}

#define RELAY_LOG(severity) LOG_IF(severity, logging_enabled())

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_LOGGING_H_
