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
#include <tvm/runtime/logging.h>

#include <string>

#if TVM_LOG_STACK_TRACE
#if TVM_USE_LIBBACKTRACE

#include <backtrace.h>
#include <cxxabi.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace tvm {
namespace runtime {
namespace {

struct BacktraceInfo {
  std::vector<std::string> lines;
  size_t max_size;
  std::string error_message;
};

void BacktraceCreateErrorCallback(void* data, const char* msg, int errnum) {
  std::cerr << "Could not initialize backtrace state: " << msg << std::endl;
}

backtrace_state* BacktraceCreate() {
  return backtrace_create_state(nullptr, 1, BacktraceCreateErrorCallback, nullptr);
}

static backtrace_state* _bt_state = BacktraceCreate();

std::string DemangleName(std::string name) {
  int status = 0;
  size_t length = name.size();
  std::unique_ptr<char, void (*)(void* __ptr)> demangled_name = {
      abi::__cxa_demangle(name.c_str(), nullptr, &length, &status), &std::free};
  if (demangled_name && status == 0 && length > 0) {
    return demangled_name.get();
  } else {
    return name;
  }
}

void BacktraceErrorCallback(void* data, const char* msg, int errnum) {
  // do nothing
}

void BacktraceSyminfoCallback(void* data, uintptr_t pc, const char* symname, uintptr_t symval,
                              uintptr_t symsize) {
  auto str = reinterpret_cast<std::string*>(data);

  if (symname != nullptr) {
    std::string tmp(symname, symsize);
    *str = DemangleName(tmp.c_str());
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

  std::unique_ptr<std::string> symbol_str = std::make_unique<std::string>("<unknown>");
  if (symbol != nullptr) {
    *symbol_str = DemangleName(symbol);
  } else {
    // see if syminfo gives anything
    backtrace_syminfo(_bt_state, pc, BacktraceSyminfoCallback, BacktraceErrorCallback,
                      symbol_str.get());
  }
  s << *symbol_str;

  if (filename != nullptr) {
    s << std::endl << "        at " << filename;
    if (lineno != 0) {
      s << ":" << lineno;
    }
  }
  // Skip tvm::backtrace and tvm::LogFatal::~LogFatal at the beginning of the trace as they don't
  // add anything useful to the backtrace.
  if (!(stack_trace->lines.size() == 0 &&
        (symbol_str->find("tvm::runtime::Backtrace", 0) == 0 ||
         symbol_str->find("tvm::runtime::detail::LogFatal", 0) == 0))) {
    stack_trace->lines.push_back(s.str());
  }
  // TVMFuncCall denotes the API boundary so we stop there. Exceptions should be caught there.
  if (*symbol_str == "TVMFuncCall" || stack_trace->lines.size() >= stack_trace->max_size) {
    return 1;
  }
  return 0;
}
}  // namespace

std::string Backtrace() {
  BacktraceInfo bt;
  bt.max_size = 100;
  if (_bt_state == nullptr) {
    return "";
  }
  // libbacktrace eats memory if run on multiple threads at the same time, so we guard against it
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  backtrace_full(_bt_state, 0, BacktraceFullCallback, BacktraceErrorCallback, &bt);

  std::ostringstream s;
  s << "Stack trace:\n";
  for (size_t i = 0; i < bt.lines.size(); i++) {
    s << "  " << i << ": " << bt.lines[i] << "\n";
  }

  return s.str();
}
}  // namespace runtime
}  // namespace tvm

#else

#include <dmlc/logging.h>

namespace tvm {
namespace runtime {
// Fallback to the dmlc implementation when backtrace is not available.
std::string Backtrace() { return dmlc::StackTrace(); }
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_USE_LIBBACKTRACE
#else

namespace tvm {
namespace runtime {
// stacktrace logging is completely disabled
std::string Backtrace() { return ""; }
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_LOG_STACK_TRACE

#if (TVM_LOG_CUSTOMIZE == 0)
namespace tvm {
namespace runtime {
namespace detail {

LogFatal::Entry& LogFatal::GetEntry() {
  static thread_local LogFatal::Entry result;
  return result;
}
}  // namespace detail
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_LOG_CUSTOMIZE
