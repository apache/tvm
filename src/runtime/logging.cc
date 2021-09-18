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
#include <unordered_map>
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
  bt.max_size = 500;
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

namespace {
constexpr const char* kSrcPrefix = "/src/";
// Note: Better would be std::char_traits<const char>::length(kSrcPrefix) but it is not
// a constexpr on all compilation targets.
constexpr const size_t kSrcPrefixLength = 5;
constexpr const char* kDefaultKeyword = "DEFAULT";
}  // namespace

// Parse \p opt_spec as a VLOG specification as per comment in
// DebugLoggingEnabled and VerboseLoggingEnabled.
std::unordered_map<std::string, int> ParseTvmLogDebugSpec(const char* opt_spec) {
  std::unordered_map<std::string, int> map;
  if (opt_spec == nullptr) {
    // DLOG and VLOG disabled.
    return map;
  }
  std::string spec(opt_spec);
  if (spec.empty() || spec == "0") {
    // DLOG and VLOG disabled.
    return map;
  }
  if (spec == "1") {
    // Legacy specification for enabling just DLOG.
    // A wildcard entry in the map will signal DLOG is on, but all VLOG levels are disabled.
    LOG(INFO) << "TVM_LOG_DEBUG enables DLOG statements only";
    map.emplace(kDefaultKeyword, -1);
    return map;
  }
  std::istringstream spec_stream(spec);
  while (spec_stream) {
    std::string name;
    if (!std::getline(spec_stream, name, '=')) {
      // Reached end.
      break;
    }
    if (name.empty()) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed, empty name";
      return map;
    }

    std::string level;
    if (!std::getline(spec_stream, level, ';')) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed, expecting level";
      return map;
    }
    if (level.empty()) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed, empty level";
      return map;
    }
    // Parse level, default to 0 if ill-formed which we don't detect.
    char* end_of_level = nullptr;
    int level_val = static_cast<int>(strtol(level.c_str(), &end_of_level, 10));
    if (end_of_level != level.c_str() + level.size()) {
      LOG(FATAL) << "TVM_LOG_DEBUG ill-formed, invalid level";
      return map;
    }
    if (map.empty()) {
      LOG(INFO) << "TVM_LOG_DEBUG enables DLOG statements";
    }
    LOG(INFO) << "TVM_LOG_DEBUG enables VLOG statements in '" << name << "' up to level " << level;
    map.emplace(name, level_val);
  }
  return map;
}

const std::unordered_map<std::string, int>& TvmLogDebugSpec() {
  // Parse and cache the verbosity level map.
  static const auto* map =
      new std::unordered_map<std::string, int>(ParseTvmLogDebugSpec(std::getenv("TVM_LOG_DEBUG")));
  return *map;
}

bool VerboseEnabledInMap(const std::string& filename, int level,
                         const std::unordered_map<std::string, int>& map) {
  if (level < 0) {
    return false;
  }
  // Canonicalize filename.
  // TODO(mbs): Not Windows friendly.

  size_t last_src = filename.rfind(kSrcPrefix, std::string::npos, kSrcPrefixLength);
  // Strip anything before the /src/ prefix, on the assumption that will yield the
  // TVM project relative filename. If no such prefix fallback to filename without
  // canonicalization.
  std::string key =
      last_src == std::string::npos ? filename : filename.substr(last_src + kSrcPrefixLength);
  // Check for exact match.
  auto itr = map.find(key);
  if (itr != map.end()) {
    return level <= itr->second;
  }
  // Check for default.
  itr = map.find(kDefaultKeyword);
  if (itr != map.end()) {
    return level <= itr->second;
  }
  return false;
}

LogFatal::Entry& LogFatal::GetEntry() {
  static thread_local LogFatal::Entry result;
  return result;
}

std::string VLogContext::str() const {
  std::stringstream result;
  for (const auto* entry : context_stack_) {
    ICHECK_NOTNULL(entry);
    result << entry->str();
    result << ": ";
  }
  return result.str();
}

}  // namespace detail
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_LOG_CUSTOMIZE
