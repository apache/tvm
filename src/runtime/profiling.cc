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

class DefaultTimerNode : public TimerNode {
  public:
    virtual void Start() {
      TVMSynchronize(ctx_.device_type, ctx_.device_id, nullptr);
      start_ = std::chrono::high_resolution_clock::now();
    };
    virtual void Stop(){
      TVMSynchronize(ctx_.device_type, ctx_.device_id, nullptr);
      duration_ = std::chrono::high_resolution_clock::now() - start_;
    };
    virtual int64_t SyncAndGetTime(){return duration_.count();};
    virtual ~DefaultTimerNode() {
    }


    DefaultTimerNode(TVMContext ctx): ctx_(ctx) {}
    static constexpr const char* _type_key = "DefaultTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(DefaultTimerNode, TimerNode);
  private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
  TVMContext ctx_;
};

TVM_REGISTER_OBJECT_TYPE(DefaultTimerNode);
TVM_REGISTER_OBJECT_TYPE(TimerNode);



Timer DefaultTimer(TVMContext ctx) {
  return Timer(GetObjectPtr<TimerNode>(new DefaultTimerNode(ctx)));
}

class CPUTimerNode : public TimerNode {
  public:
    virtual void Start() {
      start_ = std::chrono::high_resolution_clock::now();
    };
    virtual void Stop(){
      duration_ = std::chrono::high_resolution_clock::now() - start_;
    };
    virtual int64_t SyncAndGetTime(){return duration_.count();};
    virtual ~CPUTimerNode() {}


    static constexpr const char* _type_key = "CPUTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(CPUTimerNode, TimerNode);
  private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
};
TVM_REGISTER_OBJECT_TYPE(CPUTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.cpu").set_body_typed([](TVMContext ctx) {
  return Timer(GetObjectPtr<TimerNode>(new CPUTimerNode()));
});

TVM_REGISTER_GLOBAL("profiling.start_timer").set_body_typed(StartTimer);
}  // namespace runtime
}  // namespace tvm
