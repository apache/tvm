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
 *  Copyright (c) 2019 by Contributors
 * \file util.h
 * \brief Defines some common utility function..
 */
#ifndef TVM_COMMON_UTIL_H_
#define TVM_COMMON_UTIL_H_

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

namespace tvm {
namespace common {

/*!
 * \brief IsNumber check whether string is a number.
 * \param str input string
 * \return result of operation.
 */
inline bool IsNumber(const std::string& str) {
  return !str.empty() && std::find_if(str.begin(),
      str.end(), [](char c) { return !std::isdigit(c); }) == str.end();
}

/*!
 * \brief split Split the string based on delimiter
 * \param str Input string
 * \param delim The delimiter.
 * \return vector of strings which are splitted.
 */
inline std::vector<std::string> Split(const std::string& str, char delim) {
  std::string item;
  std::istringstream is(str);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

/*!
 * \brief EndsWith check whether the strings ends with
 * \param value The full string
 * \param end The end substring
 * \return bool The result.
 */
inline bool EndsWith(std::string const & value, std::string const & end) {
  if (end.size() <= value.size()) {
    return std::equal(end.rbegin(), end.rend(), value.rbegin());
  }
  return false;
}

}  // namespace common
}  // namespace tvm
#endif  // TVM_COMMON_UTIL_H_