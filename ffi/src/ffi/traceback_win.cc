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
 * \file traceback_win.cc
 * \brief Traceback implementation on windows platform
 * \note We use the term "traceback" to be consistent with python naming convention.
 */
#ifdef _MSC_VER

// clang-format off
#include <windows.h>
#include <dbghelp.h>  // NOLINT(*)
// clang-format on

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#include <iostream>
#include <vector>

#include "./traceback.h"

namespace tvm {
namespace ffi {
namespace {

std::string Traceback() {
  TracebackStorage traceback;
  HANDLE process = GetCurrentProcess();
  HANDLE thread = GetCurrentThread();

  SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
  SymInitialize(process, NULL, TRUE);
  CONTEXT context = {};
  RtlCaptureContext(&context);

  STACKFRAME64 stack = {};
  DWORD machine_type;

#if defined(_M_X64)
  machine_type = IMAGE_FILE_MACHINE_AMD64;
  stack.AddrPC.Offset = context.Rip;
  stack.AddrFrame.Offset = context.Rbp;
  stack.AddrStack.Offset = context.Rsp;
#elif defined(_M_IX86)
  machine_type = IMAGE_FILE_MACHINE_I386;
  stack.AddrPC.Offset = context.Eip;
  stack.AddrFrame.Offset = context.Ebp;
  stack.AddrStack.Offset = context.Esp;
#else
#error "Platform not supported!"
#endif

  stack.AddrPC.Mode = AddrModeFlat;
  stack.AddrFrame.Mode = AddrModeFlat;
  stack.AddrStack.Mode = AddrModeFlat;

  while (!traceback.ExceedTracebackLimit()) {
    if (!StackWalk64(machine_type, process, thread, &stack, &context, nullptr,
                     SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
      break;
    }

    if (stack.AddrPC.Offset == 0) {
      break;
    }
    const char* filename = nullptr;
    const char* symbol = "<unknown>";
    int lineno = 0;
    // Get file and line number
    IMAGEHLP_LINE64 line_info;
    ZeroMemory(&line_info, sizeof(IMAGEHLP_LINE64));
    line_info.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    DWORD displacement32 = 0;

    if (SymGetLineFromAddr64(process, stack.AddrPC.Offset, &displacement32, &line_info)) {
      filename = line_info.FileName;
      lineno = line_info.LineNumber;
    }
    // allocate symbol info that aligns to the SYMBOL_INFO
    // we use u64 here to be safe
    size_t total_symbol_bytes = sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR);
    size_t total_u64_words = (total_symbol_bytes + 7) / 8;
    static_assert(8 % alignof(SYMBOL_INFO) == 0);
    std::vector<uint64_t> symbol_buffer(total_u64_words, 0);
    PSYMBOL_INFO symbol_info = reinterpret_cast<PSYMBOL_INFO>(symbol_buffer.data());
    symbol_info->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol_info->MaxNameLen = MAX_SYM_NAME;
    DWORD64 displacement = 0;
    if (SymFromAddr(process, stack.AddrPC.Offset, &displacement, symbol_info)) {
      symbol = symbol_info->Name;
    }

    if (ShouldStopTraceback(filename, symbol)) {
      break;
    }
    if (ShouldExcludeFrame(filename, symbol)) {
      continue;
    }
    traceback.Append(filename, symbol, lineno);
  }
  SymCleanup(process);
  return traceback.GetTraceback();
}
}  // namespace
}  // namespace ffi
}  // namespace tvm

const TVMFFIByteArray* TVMFFITraceback(const char* filename, int lineno, const char* func) {
  static thread_local std::string traceback_str;
  static thread_local TVMFFIByteArray traceback_array;
  traceback_str = ::tvm::ffi::Traceback();
  traceback_array.data = traceback_str.data();
  traceback_array.size = traceback_str.size();
  return &traceback_array;
}
#endif  // _MSC_VER
