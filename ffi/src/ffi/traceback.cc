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
 * \file traceback.cc
 * \brief Traceback implementation on non-windows platforms
 * \note We use the term "traceback" to be consistent with python naming convention.
 */
#ifndef _MSC_VER

#include "./traceback.h"

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#if TVM_FFI_USE_LIBBACKTRACE

#include <backtrace.h>
#include <cxxabi.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>

#if TVM_FFI_BACKTRACE_ON_SEGFAULT
#include <csignal>
#endif

namespace tvm {
namespace ffi {
namespace {

void BacktraceCreateErrorCallback(void*, const char* msg, int) {
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

void BacktraceErrorCallback(void*, const char*, int) {
  // do nothing
}

void BacktraceSyminfoCallback(void* data, uintptr_t pc, const char* symname, uintptr_t,
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
  auto stack_trace = reinterpret_cast<TracebackStorage*>(data);
  std::string symbol_str = "<unknown>";
  if (symbol) {
    symbol_str = DemangleName(symbol);
  } else {
    // see if syminfo gives anything
    backtrace_syminfo(_bt_state, pc, BacktraceSyminfoCallback, BacktraceErrorCallback, &symbol_str);
  }
  symbol = symbol_str.data();

  if (stack_trace->ExceedTracebackLimit()) {
    return 1;
  }
  if (ShouldExcludeFrame(filename, symbol)) {
    return 0;
  }
  stack_trace->Append(filename, symbol, lineno);
  return 0;
}

std::string Traceback() {
  TracebackStorage traceback;

  if (_bt_state == nullptr) {
    return "";
  }
  // libbacktrace eats memory if run on multiple threads at the same time, so we guard against it
  {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    backtrace_full(_bt_state, 0, BacktraceFullCallback, BacktraceErrorCallback, &traceback);
  }
  return traceback.GetTraceback();
}

#if TVM_FFI_BACKTRACE_ON_SEGFAULT
void backtrace_handler(int sig) {
  // Technically we shouldn't do any allocation in a signal handler, but
  // Backtrace may allocate. What's the worst it could do? We're already
  // crashing.
  std::cerr << "!!!!!!! TVM FFI encountered a Segfault !!!!!!!\n" << Traceback() << std::endl;

  // Re-raise signal with default handler
  struct sigaction act;
  std::memset(&act, 0, sizeof(struct sigaction));
  act.sa_flags = SA_RESETHAND;
  act.sa_handler = SIG_DFL;
  sigaction(sig, &act, nullptr);
  raise(sig);
}

__attribute__((constructor)) void install_signal_handler(void) {
  // this may override already installed signal handlers
  std::signal(SIGSEGV, backtrace_handler);
}
#endif  // TVM_FFI_BACKTRACE_ON_SEGFAULT
}  // namespace
}  // namespace ffi
}  // namespace tvm

const char* TVMFFITraceback(const char*, int, const char*) {
  static thread_local std::string traceback_str;
  traceback_str = ::tvm::ffi::Traceback();
  return traceback_str.c_str();
}
#else
// fallback implementation simply print out the last trace
const char* TVMFFITraceback(const char* filename, int lineno, const char* func) {
  static thread_local std::string traceback_str;
  std::ostringstream traceback_stream;
  // python style backtrace
  traceback_stream << "  File \"" << filename << "\", line " << lineno << ", in " << func << '\n';
  traceback_str = traceback_stream.str();
  return traceback_str.c_str();
}
#endif  // TVM_FFI_USE_LIBBACKTRACE
#endif  // _MSC_VER
