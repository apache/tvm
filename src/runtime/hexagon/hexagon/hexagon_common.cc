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
 * \file hexagon_common.cc
 */

#include "hexagon_common.h"
#include <tvm/runtime/logging.h>

#include <string>
#include <sstream>

namespace {
std::vector<std::string> SplitString(const std::string& str, char delim) {
  std::vector<std::string> lines;
  auto ss = std::stringstream{str};
  for (std::string line; std::getline(ss, line, delim);) {
    lines.push_back(line);
  }
  return lines;
}
void HexagonLog(const std::string& file, int lineno, const std::string& message) {
  HEXAGON_PRINT(ALWAYS, "%s:%d:", file.c_str(), lineno);
  std::vector<std::string> err_lines = SplitString(message, '\n');
  for (auto& line : err_lines) {
    HEXAGON_PRINT(ALWAYS, "%s", line.c_str());
  }
}
}  // namespace

namespace tvm {
namespace runtime {
namespace detail {
void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  HexagonLog(file, lineno, message);
  throw InternalError(file, lineno, message);
}
void LogMessageImpl(const std::string& file, int lineno, const std::string& message) {
  HexagonLog(file, lineno, message);
}

}  // namespace detail
}  // namespace runtime
}  // namespace tvm
