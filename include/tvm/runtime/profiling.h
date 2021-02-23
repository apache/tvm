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

/*! \brief Base class for all implementations.
 *
 * New implementations of this interface should make sure that `Start` and `Stop`
 * are as lightweight as possible. Expensive state synchronization should be
 * done in `SyncAndGetTime`.
 */
class TimerNode : public Object {
 public:
  /*! \brief Start the timer.
   *
   * Note: this function should only be called once per object.
   */
  virtual void Start() = 0;
  /*! \brief Stop the timer.
   *
   * Note: this function should only be called once per object.
   */
  virtual void Stop() = 0;
  /*! \brief Synchronize timer state and return elapsed time between `Start` and `Stop`.
   * \return The time in nanoseconds between `Start` and `Stop`.
   *
   * This function is necessary because we want to avoid timing the overhead of
   * doing timing. When using multiple timers, it is recommended to stop all of
   * them before calling `SyncAndGetTime` on any of them.
   *
   * Note: this function should be only called once per object. It may incur
   * a large synchronization overhead (for example, with GPUs).
   */
  virtual int64_t SyncAndGetTime() = 0;

  virtual ~TimerNode() {}

  static constexpr const char* _type_key = "TimerNode";
  TVM_DECLARE_BASE_OBJECT_INFO(TimerNode, Object);
};

/*! \brief Timer for a specific device.
 *
 * You should not construct this class directly. Instead use `StartTimer`.
 */
class Timer : public ObjectRef {
 public:
  /*! \brief Start the timer.
   *
   * Note: this function should only be called once per object.
   */
  void Start() { operator->()->Start(); }
  /*! \brief Stop the timer.
   *
   * Note: this function should only be called once per object.
   */
  void Stop() { operator->()->Stop(); }
  /*! \brief Synchronize timer state and return elapsed time between `Start` and `Stop`.
   * \return The time in nanoseconds between `Start` and `Stop`.
   *
   * This function is necessary because we want to avoid timing the overhead of
   * doing timing. When using multiple timers, it is recommended to stop all of
   * them before calling `SyncAndGetTime` on any of them.
   *
   * Note: this function should be only called once per object. It may incur
   * a large synchronization overhead (for example, with GPUs).
   */
  int64_t SyncAndGetTime() { return operator->()->SyncAndGetTime(); }
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
 * \return A `Timer` that has already been started.
 *
 * Example usage:
 * \code{.cpp}
 * Timer t = StartTimer(TVMContext::cpu());
 * my_long_running_function();
 * t.Stop();
 * ... // some more computation
 * int64_t nanosecs = t.SyncAndGetTime() // elapsed time in nanoseconds
 * \endcode
 *
 * To add a new device-specific timer, register a new function
 * "profiler.timer.my_device" (where `my_device` is the `DeviceName` of your
 * device). This function should accept a `TVMContext` and return a new `Timer`
 * that has already been started.
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
