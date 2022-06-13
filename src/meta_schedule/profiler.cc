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
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

/**************** Context Manager ****************/

class ProfilerInternal {
 public:
  static void EnterScope(Profiler ctx) { ctx.EnterWithScope(); }
  static void ExitScope(Profiler ctx) { ctx.ExitWithScope(); }
};

void Profiler::EnterWithScope() {
  Optional<Profiler>& ctx = ProfilerThreadLocalStore::Get()->ctx;
  CHECK(!ctx.defined()) << "ValueError: Nested Profiler context managers are not allowed";
  ctx = *this;
}

void Profiler::ExitWithScope() {
  Optional<Profiler>& ctx = ProfilerThreadLocalStore::Get()->ctx;
  ICHECK(ctx.defined());
  ctx = NullOpt;
}

/**************** Profiler ****************/

Profiler::Profiler() {
  ObjectPtr<ProfilerNode> n = make_object<ProfilerNode>();
  data_ = n;
}

ScopedTimer ProfilerNode::TimeScope(String name) {
  return ScopedTimer([name, tick = std::chrono::high_resolution_clock::now()]() -> void {
    Optional<Profiler> profiler = ProfilerThreadLocalStore::Get()->ctx;
    if (profiler.defined()) {
      Map<String, FloatImm>& stats = profiler.value()->stats;
      double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now() - tick)
                            .count() /
                        1e9 / 60;
      if (stats.find(name) != stats.end()) {
        stats.Set(name, FloatImm(DataType::Float(64), stats.at(name)->value + duration));
      } else {
        stats.Set(name, FloatImm(DataType::Float(64), duration));
      }
    }
  });
}

void ProfilerNode::StartContextTimer(String name) {
  stack.push_back(std::make_pair(name, std::chrono::high_resolution_clock::now()));
}

void ProfilerNode::EndContextTimer() {
  ICHECK(stack.size() > 0) << "There is no timer context running!";
  String name = stack.back().first;
  double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now() - stack.back().second)
                        .count() /
                    1e9 / 60;
  if (stats.find(name) != stats.end()) {
    stats.Set(name, FloatImm(DataType::Float(64), stats.at(name)->value + duration));
  } else {
    stats.Set(name, FloatImm(DataType::Float(64), duration));
  }
  stack.pop_back();
}

TVM_REGISTER_NODE_TYPE(ProfilerNode);
TVM_REGISTER_GLOBAL("meta_schedule.Profiler").set_body_typed([]() -> Profiler {
  return Profiler();
});
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerEnterScope")
    .set_body_typed(ProfilerInternal::EnterScope);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerExitScope").set_body_typed(ProfilerInternal::ExitScope);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerStartContextTimer")
    .set_body_method<Profiler>(&ProfilerNode::StartContextTimer);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerEndContextTimer")
    .set_body_method<Profiler>(&ProfilerNode::EndContextTimer);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerGet").set_body_method<Profiler>(&ProfilerNode::Get);

}  // namespace meta_schedule
}  // namespace tvm
