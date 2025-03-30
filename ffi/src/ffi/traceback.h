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
 * \file traceback.h
 * \brief Common headers for traceback.
 * \note We use the term "traceback" to be consistent with python naming convention.
 */
#ifndef TVM_FFI_TRACEBACK_H_
#define TVM_FFI_TRACEBACK_H_

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace ffi {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // std::getenv is unsafe
#endif

inline int32_t GetTracebackLimit() {
  if (const char* env = std::getenv("TVM_TRACEBACK_LIMIT")) {
    return std::stoi(env);
  }
  return 512;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/*!
 * \brief List frame patterns that should be excluded as they contain less information
 */
inline bool ShouldExcludeFrame(const char* filename, const char* symbol) {
  if (filename) {
    // Stack frames for TVM FFI
    if (strstr(filename, "include/tvm/ffi/error.h")) {
      return true;
    }
    if (strstr(filename, "include/tvm/ffi/function_details.h")) {
      return true;
    }
    if (strstr(filename, "include/tvm/ffi/function.h")) {
      return true;
    }
    if (strstr(filename, "include/tvm/ffi/any.h")) {
      return true;
    }
    if (strstr(filename, "src/ffi/traceback.cc")) {
      return true;
    }
    // Python interpreter stack frames
    if (strstr(filename, "/python-") || strstr(filename, "/Python/ceval.c") ||
        strstr(filename, "/Modules/_ctypes")) {
      return true;
    }
    // C++ stdlib frames
    if (strstr(filename, "include/c++/")) {
      return true;
    }
  }

  if (symbol) {
    // C++ stdlib frames
    if (strstr(symbol, "__libc_")) {
      return true;
    }
    // Python interpreter stack frames
    if (strstr(symbol, "_Py") == symbol || strstr(symbol, "PyObject")) {
      return true;
    }
  }
  // libffi.so stack frames.  These may also show up as numeric
  // addresses with no symbol name.  This could be improved in the
  // future by using dladdr() to check whether an address is contained
  // in libffi.so
  if (strstr(symbol, "ffi_call_")) {
    return true;
  }
  return false;
}

/*!
 * \brief storage to store traceback
 */
struct TracebackStorage {
  std::vector<std::string> lines;
  /*! \brief Maximum size of the traceback. */
  size_t max_frame_size = GetTracebackLimit();

  void Append(const char* filename, const char* func, int lineno) {
    // skip frames with empty filename
    if (filename == nullptr) return;
    std::ostringstream trackeback_stream;
    trackeback_stream << "  " << filename;
    if (lineno != 0) {
      trackeback_stream << ", line " << lineno;
    }
    trackeback_stream << ", in " << func << '\n';
    lines.push_back(trackeback_stream.str());
  }

  bool ExceedTracebackLimit() const { return lines.size() >= max_frame_size; }

  // get traceback in the order of most recent call last
  std::string GetTraceback() const {
    std::string traceback;
    for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
      traceback.insert(traceback.end(), it->begin(), it->end());
    }
    return traceback;
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_TRACEBACK_H_
