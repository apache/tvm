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

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/profiling.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "HAP_debug.h"
#include "HAP_perf.h"
#include "hexagon_buffer.h"

namespace tvm {
namespace runtime {
namespace hexagon {

class HexagonTimerNode : public TimerNode {
 public:
  virtual void Start() { start = HAP_perf_get_time_us(); }
  virtual void Stop() { end = HAP_perf_get_time_us(); }
  virtual int64_t SyncAndGetElapsedNanos() { return (end - start) * 1e3; }
  virtual ~HexagonTimerNode() {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.hexagon.HexagonTimerNode", HexagonTimerNode,
                                    TimerNode);

 private:
  uint64_t start, end;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("profiling.timer.hexagon",
                        [](Device dev) { return Timer(ffi::make_object<HexagonTimerNode>()); });
}
}  // namespace hexagon

namespace {
std::vector<std::string> SplitString(const std::string& str, char delim) {
  std::vector<std::string> lines;
  auto ss = std::stringstream{str};
  for (std::string line; std::getline(ss, line, delim);) {
    lines.push_back(line);
  }
  return lines;
}
void HexagonLog(const std::string& file, int lineno, int level, const std::string& message) {
  std::vector<std::string> err_lines = SplitString(message, '\n');
  for (auto& line : err_lines) {
    // TVM log levels roughly map to HAP log levels
    HAP_debug_runtime(level, file.c_str(), lineno, line.c_str());
  }
}
}  // namespace

namespace detail {
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  HexagonLog(file, lineno, TVM_LOG_LEVEL_FATAL, message);
  throw InternalError(file, lineno, message);
}
void LogMessageImpl(const std::string& file, int lineno, int level, const std::string& message) {
  HexagonLog(file, lineno, level, message);
}
}  // namespace detail

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "ffi.Module.load_from_file.hexagon", [](ffi::PackedArgs args, ffi::Any* rv) {
        auto floader = tvm::ffi::Function::GetGlobalRequired("ffi.Module.load_from_file.so");
        *rv = floader(args[0].cast<ffi::String>(), "so");
      });
}

}  // namespace runtime
}  // namespace tvm
