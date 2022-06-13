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

std::vector<Profiler>* ThreadLocalProfilers() {
  static thread_local std::vector<Profiler> profilers;
  return &profilers;
}

void Profiler::EnterWithScope() { ThreadLocalProfilers()->push_back(*this); }
void Profiler::ExitWithScope() { ThreadLocalProfilers()->pop_back(); }

Optional<Profiler> Profiler::Current() {
  std::vector<Profiler>* profilers = ThreadLocalProfilers();
  if (profilers->empty()) {
    return NullOpt;
  } else {
    return profilers->back();
  }
}

/**************** Profiler ****************/

Map<String, FloatImm> ProfilerNode::Get() const {
  Map<String, FloatImm> ret;
  for (const auto& kv : stats_sec) {
    ret.Set(kv.first, FloatImm(DataType::Float(64), kv.second));
  }
  return ret;
}

String ProfilerNode::Table() const {
  // TODO(@zxybazh): Add a table format.
}

Profiler::Profiler() {
  ObjectPtr<ProfilerNode> n = make_object<ProfilerNode>();
  n->stats_sec.clear();
  data_ = n;
}

runtime::TypedPackedFunc<void()> ProfilerTimedScope(String name) {
  if (Optional<Profiler> opt_profiler = Profiler::Current()) {
    return [profiler = opt_profiler.value(),                  //
            tik = std::chrono::high_resolution_clock::now(),  //
            name = std::move(name)]() {
      auto tok = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration_cast<std::chrono::seconds>(tok - tik).count();
      profiler->stats_sec[name] += duration;
    };
  }
  return nullptr;
}

ScopedTimer Profiler::TimedScope(String name) { return ScopedTimer(ProfilerTimedScope(name)); }

TVM_REGISTER_NODE_TYPE(ProfilerNode);
TVM_REGISTER_GLOBAL("meta_schedule.Profiler").set_body_typed([]() -> Profiler {
  return Profiler();
});
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerEnterWithScope")
    .set_body_method(&Profiler::EnterWithScope);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerExitWithScope")
    .set_body_method(&Profiler::ExitWithScope);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerCurrent").set_body_typed(Profiler::Current);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerTimedScope").set_body_typed(ProfilerTimedScope);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerGet").set_body_method<Profiler>(&ProfilerNode::Get);
TVM_REGISTER_GLOBAL("meta_schedule.ProfilerTable").set_body_method<Profiler>(&ProfilerNode::Table);

}  // namespace meta_schedule
}  // namespace tvm
