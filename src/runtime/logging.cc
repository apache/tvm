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

#ifdef TVM_BACKTRACE_DISABLED
#include <string>

// TODO(bkimball,tkonolige) This inline function is to work around a linking error I am having when
// using MSVC If the function definition is in logging.cc then the linker can't find it no matter
// what kind of attributes (dllexport) I decorate it with. This is temporary and will be addressed
// when we get backtrace working on Windows.
namespace tvm {
namespace runtime {
TVM_DLL std::string Backtrace() { return ""; }
}  // namespace runtime
}  // namespace tvm
#else

#include <backtrace.h>
#include <cxxabi.h>
#include <tvm/runtime/logging.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace {

struct BacktraceInfo {
  std::vector<std::string> lines;
  size_t max_size;
  std::string error_message;
};

backtrace_state** GetBacktraceState() {
  thread_local backtrace_state* state = nullptr;
  return &state;
}

std::string DemangleName(const char* name) {
  int status = 0;
  size_t length = std::string::npos;
  std::unique_ptr<char, void (*)(void* __ptr)> demangled_name = {
      abi::__cxa_demangle(name, nullptr, &length, &status), &std::free};
  if (demangled_name && status == 0 && length > 0) {
    return demangled_name.get();
  } else {
    return name;
  }
}

void BacktraceErrorCallback(void* data, const char* msg, int errnum) {
  auto stack_trace = reinterpret_cast<BacktraceInfo*>(data);
  stack_trace->error_message = msg;
}

void BacktraceSyminfoCallback(void* data, uintptr_t pc, const char* symname, uintptr_t symval,
                              uintptr_t symsize) {
  auto str = reinterpret_cast<std::string*>(data);

  if (symname != nullptr) {
    *str = DemangleName(symname);
  } else {
    std::ostringstream s;
    s << "0x" << std::setfill('0') << std::setw(sizeof(uintptr_t) * 2) << std::hex << pc;
    *str = s.str();
  }
}
int BacktraceFullCallback(void* data, uintptr_t pc, const char* filename, int lineno,
                          const char* symbol) {
  auto stack_trace = reinterpret_cast<BacktraceInfo*>(data);
  std::stringstream s;

  std::string symbol_str = "<unknown>";
  if (symbol != nullptr) {
    symbol_str = DemangleName(symbol);
  } else {
    // see if syminfo gives anything
    backtrace_state** bt_state = GetBacktraceState();
    backtrace_syminfo(*bt_state, pc, BacktraceSyminfoCallback, BacktraceErrorCallback, &symbol_str);
  }
  s << symbol_str;

  if (filename != nullptr) {
    s << std::endl << "        at " << filename;
    if (lineno != 0) {
      s << ":" << lineno;
    }
  }
  // Skip tvm::backtrace and tvm::LogFatal::~LogFatal at the beginning of the trace as they don't
  // add anything useful to the backtrace.
  if (!(stack_trace->lines.size() == 0 &&
        (symbol_str.find("tvm::runtime::Backtrace", 0) == 0 ||
         symbol_str.find("tvm::runtime::detail::LogFatal", 0) == 0))) {
    stack_trace->lines.push_back(s.str());
  }
  // TVMFuncCall denotes the API boundary so we stop there. Exceptions should be caught there.
  if (symbol_str == "TVMFuncCall" || stack_trace->lines.size() >= stack_trace->max_size) {
    return 1;
  }
  return 0;
}
}  // namespace

std::string Backtrace() {
  BacktraceInfo bt;
  bt.max_size = 100;
  backtrace_state** bt_state = GetBacktraceState();
  if (*bt_state == nullptr) {
    // TVM is loaded as a shared library into python, so we cannot supply an
    // executable path to libbacktrace.
    *bt_state = backtrace_create_state(nullptr, 1, BacktraceErrorCallback, &bt);
  }
  backtrace_full(*bt_state, 0, BacktraceFullCallback, BacktraceErrorCallback, &bt);

  std::ostringstream s;
  s << "Stack trace:\n";
  for (size_t i = 0; i < bt.lines.size(); i++) {
    s << "  " << i << ": " << bt.lines[i] << "\n";
  }

  return s.str();
}
}  // namespace runtime
}  // namespace tvm
#endif
