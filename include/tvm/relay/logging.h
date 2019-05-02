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

/*!
 * \file tvm/relay/logging.h
 * \brief A wrapper around dmlc-core/logging.h which adds the ability
 * to toggle logging via an environment variable.
 */

#ifndef TVM_RELAY_LOGGING_H_
#define TVM_RELAY_LOGGING_H_

#include <dmlc/logging.h>
#include <tvm/logging.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

namespace tvm {
namespace relay {

#ifdef USE_RELAY_DEBUG
struct EnableRelayDebug {
  static constexpr bool value = true;
};

static inline bool logging_enabled() {
  if (auto var = std::getenv("USE_RELAY_LOG")) {
    std::string is_on(var);
    return is_on == "1";
  } else {
    return false;
  }
}

// Use dmlc logging directly if debugging mode is enabled.
#define RELAY_LOG(severity) LOG_IF(severity, ::tvm::relay::logging_enabled())

// Define various Relay assertion helpers.
#define RELAY_ASSERT(condition) CHECK(condition)
#define RELAY_ASSERT_EQ(val1, val2) CHECK_EQ(val1, val2)
#define RELAY_ASSERT_NE(val1, val2) CHECK_NE(val1, val2)
#define RELAY_ASSERT_LE(val1, val2) CHECK_LE(val1, val2)
#define RELAY_ASSERT_LT(val1, val2) CHECK_LT(val1, val2)
#define RELAY_ASSERT_GE(val1, val2) CHECK_GE(val1, val2)
#define RELAY_ASSERT_GT(val1, val2) CHECK_GT(val1, val2)

#else
struct EnableRelayDebug {
  static constexpr bool value = false;
};

// Define an empty class that will ignore error messages during compile-time.
class RelayLog {};
static const RelayLog relay_log;

template <typename T>
static const inline RelayLog& operator<<(const RelayLog& log, const T& msg) {
  return log;
}

// The signature for std::endl manipulator.
using EndlManipulator = std::add_pointer<std::ostream&(std::ostream&)>::type;

// Define an operator<< to take std::endl as well.
static const inline RelayLog& operator<<(const RelayLog& log,
                                         const EndlManipulator& msg) {
  return log;
}

#define RELAY_LOG(severity) ::tvm::relay::relay_log
#define RELAY_ASSERT(condition) ::tvm::relay::relay_log
#define RELAY_ASSERT_EQ(val1, val2) ::tvm::relay::relay_log
#define RELAY_ASSERT_NE(val1, val2) ::tvm::relay::relay_log
#define RELAY_ASSERT_LE(val1, val2) ::tvm::relay::relay_log
#define RELAY_ASSERT_LT(val1, val2) ::tvm::relay::relay_log
#define RELAY_ASSERT_GE(val1, val2) ::tvm::relay::relay_log
#define RELAY_ASSERT_GT(val1, val2) :tvm::relay::relay_log
#endif  // USE_RELAY_DEBUG

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_LOGGING_H_
