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
 * \file src/runtime/profiling.cc
 * \brief Runtime profiling including timers.
 */

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>

namespace tvm {
namespace runtime {
TypedPackedFunc<int64_t()> DefaultTimer(TVMContext ctx) {
  TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
  auto start = std::chrono::steady_clock::now();
  return TypedPackedFunc<int64_t()>(
      [=]() -> int64_t {
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
        auto stop = std::chrono::steady_clock::now();
        std::chrono::duration<int64_t, std::nano> duration = stop - start;
        return duration.count();
      },
      "profiling.timer.default.stop");
}

TVM_REGISTER_GLOBAL("profiling.timer.cpu").set_body_typed([](TVMContext ctx) {
  auto start = std::chrono::steady_clock::now();
  return TypedPackedFunc<int64_t()>(
      [=]() -> int64_t {
        auto stop = std::chrono::steady_clock::now();
        std::chrono::duration<int64_t, std::nano> duration = stop - start;
        return duration.count();
      },
      "profiling.timer.cpu.stop");
});

TVM_REGISTER_GLOBAL("profiling.start_timer").set_body_typed(StartTimer);
}  // namespace runtime
}  // namespace tvm
