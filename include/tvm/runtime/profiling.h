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
 * \file include/tvm/runtime/profiling.h
 * \brief Runtime profiling including timers.
 */
#ifndef TVM_RUNTIME_PROFILING_H_
#define TVM_RUNTIME_PROFILING_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <map>
#include <string>

namespace tvm {
namespace runtime {

class TimerNode : public Object {
  public:
    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual int64_t SyncAndGetTime() = 0;
    virtual ~TimerNode() {};

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const uint32_t _type_child_slots = 4;
  static constexpr const uint32_t _type_child_slots_can_overflow = true;
    static constexpr const char* _type_key = "TimerNode";
  TVM_DECLARE_BASE_OBJECT_INFO(TimerNode, Object);
};

class Timer : public ObjectRef {
  public:
    void Start() {
      operator->()->Start();
    }
    void Stop() {
      operator->()->Stop();
    }
    int64_t SyncAndGetTime() {
      return operator->()->SyncAndGetTime();
    }
    TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Timer, ObjectRef, TimerNode);
};

/*!
 * \brief Default timer if one does not exist to the platform.
 * \param ctx The context to time on.
 *
 * Note that this timer performs synchronization between the device and CPU,
 * which can lead to overhead in the reported results.
 */
Timer DefaultTimer(TVMContext ctx);

/*!
 * \brief Get a device specific timer.
 * \param ctx The device context to time.
 * \return A function, that when called starts a timer. The results from this
 *         function is another function that will stop the timer and return
 *         another function that returns the elapsed time in nanoseconds. The
 *         third function should be called as late as possible to avoid
 *         synchronization overhead.
 *
 * This three function approach is complicated, but it is necessary to avoid
 * synchronization overhead on GPUs. On GPUs, the first two function generate
 * start and stop events respectively. The third function synchronizes the GPU
 * with the CPU and gets the elapsed time between events.
 *
 * Users can register a timer for a device by registering a packed function
 * with the name "profiler.timer.device_name". This function should take a
 * TVMContext and return a new function. The new function will stop the timer
 * when called and returns a third function. The third function should return
 * the elapsed time between the first and second call in nanoseconds.
 *
 * Note that timers are specific to a context (and by extension device stream).
 * The code being timed should run on the specific context only, otherwise you
 * may get mixed results. Furthermore, the context should not be modified
 * between the start and end of the timer (i.e. do not call TVMDeviceSetStream).
 *
 * Example usage:
 * \code{.cpp}
 * Timer t = StartTimer(TVMContext::cpu());
 * my_long_running_function();
 * t.Stop();
 * ... // some more computation
 * int64_t nanosecs = t.SyncAndGetTime() // elapsed time in nanoseconds
 * \endcode
 */
inline Timer StartTimer(TVMContext ctx) {
  auto f = Registry::Get(std::string("profiling.timer.") + DeviceName(ctx.device_type));
  if (f == nullptr) {
    Timer t = DefaultTimer(ctx);
    t.Start();
    return t;
  } else {
    Timer t = f->operator()(ctx);
    t.Start();
    return t;
  }
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_PROFILING_H_
