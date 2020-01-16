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
 * \file util.h
 * \brief Defines some common utility function..
 */
#ifndef TVM_SUPPORT_UTIL_H_
#define TVM_SUPPORT_UTIL_H_

#include <stdio.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <sys/types.h>
#endif
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <array>
#include <cctype>
#include <memory>

namespace tvm {
namespace support {
/*!
 * \brief TVMPOpen wrapper of popen between windows / unix.
 * \param command executed command
 * \param type "r" is for reading or "w" for writing.
 * \return normal standard stream
 */
inline FILE* TVMPOpen(const char* command, const char* type) {
#if defined(_WIN32)
  return _popen(command, type);
#else
  return popen(command, type);
#endif
}

/*!
 * \brief TVMPClose wrapper of pclose between windows / linux
 * \param stream the stream needed to be close.
 * \return exit status
 */
inline int TVMPClose(FILE* stream) {
#if defined(_WIN32)
  return _pclose(stream);
#else
  return pclose(stream);
#endif
}

/*!
 * \brief TVMWifexited wrapper of WIFEXITED between windows / linux
 * \param status The status field that was filled in by the wait or waitpid function
 * \return the exit code of the child process
 */
inline int TVMWifexited(int status) {
#if defined(_WIN32)
  return (status != 3);
#else
  return WIFEXITED(status);
#endif
}

/*!
 * \brief TVMWexitstatus wrapper of WEXITSTATUS between windows / linux
 * \param status The status field that was filled in by the wait or waitpid function.
 * \return the child process exited normally or not
 */
inline int TVMWexitstatus(int status) {
#if defined(_WIN32)
  return status;
#else
  return WEXITSTATUS(status);
#endif
}


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
inline bool EndsWith(std::string const& value, std::string const& end) {
  if (end.size() <= value.size()) {
    return std::equal(end.rbegin(), end.rend(), value.rbegin());
  }
  return false;
}

/*!
 * \brief Execute the command
 * \param cmd The command we want to execute
 * \param err_msg The error message if we have
 * \return executed output status
 */
inline int Execute(std::string cmd, std::string* err_msg) {
  std::array<char, 128> buffer;
  std::string result;
  cmd += " 2>&1";
  FILE* fd = TVMPOpen(cmd.c_str(), "r");
  while (fgets(buffer.data(), buffer.size(), fd) != nullptr) {
    *err_msg += buffer.data();
  }
  int status = TVMPClose(fd);
  if (TVMWifexited(status)) {
    return TVMWexitstatus(status);
  }
  return 255;
}

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_UTIL_H_
