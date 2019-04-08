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
