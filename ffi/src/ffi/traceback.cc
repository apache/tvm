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
  char* demangled_name = abi::__cxa_demangle(name.c_str(), nullptr, &length, &status);
  if (demangled_name && status == 0 && length > 0) {
    name = demangled_name;
  }
  if (demangled_name) {
    std::free(demangled_name);
  }
  return name;
}

void BacktraceErrorCallback(void*, const char*, int) {
  // do nothing
}

void BacktraceSyminfoCallback(void* data, uintptr_t pc, const char* symname, uintptr_t, uintptr_t) {
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
  if (stack_trace->stop_at_boundary && DetectFFIBoundary(filename, symbol)) {
    return 1;
  }
  // skip extra frames
  if (stack_trace->skip_frame_count > 0) {
    stack_trace->skip_frame_count--;
    return 0;
  }
  if (ShouldExcludeFrame(filename, symbol)) {
    return 0;
  }
  stack_trace->Append(filename, symbol, lineno);
  return 0;
}
}  // namespace
}  // namespace ffi
}  // namespace tvm

const TVMFFIByteArray* TVMFFITraceback(const char* filename, int lineno, const char* func,
                                       int cross_ffi_boundary) {
  // We collapse the traceback into a single function
  // to simplify the traceback detection handling (since we need to detect TVMFFITraceback)
  static thread_local std::string traceback_str;
  static thread_local TVMFFIByteArray traceback_array;
  // pass in current line as here so last line of traceback is always accurate
  tvm::ffi::TracebackStorage traceback;
  traceback.stop_at_boundary = cross_ffi_boundary == 0;
  if (filename != nullptr && func != nullptr) {
    // need to skip TVMFFITraceback and the caller function
    // which is already included in filename and func
    traceback.skip_frame_count = 2;
    if (!tvm::ffi::ShouldExcludeFrame(filename, func)) {
      traceback.Append(filename, func, lineno);
    }
  }
  // libbacktrace eats memory if run on multiple threads at the same time, so we guard against it
  if (tvm::ffi::_bt_state != nullptr) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    backtrace_full(tvm::ffi::_bt_state, 0, tvm::ffi::BacktraceFullCallback,
                   tvm::ffi::BacktraceErrorCallback, &traceback);
  }
  traceback_str = traceback.GetTraceback();
  traceback_array.data = traceback_str.data();
  traceback_array.size = traceback_str.size();
  return &traceback_array;
}

#if TVM_FFI_BACKTRACE_ON_SEGFAULT
void TVMFFISegFaultHandler(int sig) {
  // Technically we shouldn't do any allocation in a signal handler, but
  // Backtrace may allocate. What's the worst it could do? We're already
  // crashing.
  const TVMFFIByteArray* traceback = TVMFFITraceback(nullptr, 0, nullptr, 1);
  std::cerr << "!!!!!!! Segfault encountered !!!!!!!\n"
            << std::string(traceback->data, traceback->size) << std::endl;
  // Re-raise signal with default handler
  struct sigaction act;
  std::memset(&act, 0, sizeof(struct sigaction));
  act.sa_flags = SA_RESETHAND;
  act.sa_handler = SIG_DFL;
  sigaction(sig, &act, nullptr);
  raise(sig);
}

__attribute__((constructor)) void TVMFFIInstallSignalHandler(void) {
  // this may override already installed signal handlers
  std::signal(SIGSEGV, TVMFFISegFaultHandler);
}
#endif  // TVM_FFI_BACKTRACE_ON_SEGFAULT
#else
// fallback implementation simply print out the last trace
const TVMFFIByteArray* TVMFFITraceback(const char* filename, int lineno, const char* func,
                                       int cross_ffi_boundary) {
  static thread_local std::string traceback_str;
  static thread_local TVMFFIByteArray traceback_array;
  std::ostringstream traceback_stream;
  if (filename != nullptr && func != nullptr) {
    // python style backtrace
    traceback_stream << "  File \"" << filename << "\", line " << lineno << ", in " << func << '\n';
  }
  traceback_str = traceback_stream.str();
  traceback_array.data = traceback_str.data();
  traceback_array.size = traceback_str.size();
  return &traceback_array;
}
#endif  // TVM_FFI_USE_LIBBACKTRACE
#endif  // _MSC_VER
