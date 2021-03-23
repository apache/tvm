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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <string>

namespace tvm {
namespace runtime {

/*! \brief Base class for all implementations.
 *
 * New implementations of this interface should make sure that `Start` and `Stop`
 * are as lightweight as possible. Expensive state synchronization should be
 * done in `SyncAndGetElapsedNanos`.
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
   * them before calling `SyncAndGetElapsedNanos` on any of them.
   *
   * Note: this function should be only called once per object. It may incur
   * a large synchronization overhead (for example, with GPUs).
   */
  virtual int64_t SyncAndGetElapsedNanos() = 0;

  virtual ~TimerNode() {}

  static constexpr const char* _type_key = "TimerNode";
  TVM_DECLARE_BASE_OBJECT_INFO(TimerNode, Object);
};

/*! \brief Timer for a specific device.
 *
 * This is a managed reference to a TimerNode.
 *
 * \sa TimerNode
 */
class Timer : public ObjectRef {
 public:
  /*!
   * \brief Get a device specific timer.
   * \param dev The device to time.
   * \return A `Timer` that has already been started.
   *
   * Use this function to time runtime of arbitrary regions of code on a specific
   * device. The code that you want to time should be running on the device
   * otherwise the timer will not return correct results. This is a lower level
   * interface than TimeEvaluator and only runs the timed code once
   * (TimeEvaluator runs the code multiple times).
   *
   * A default timer is used if a device specific one does not exist. This
   * timer performs synchronization between the device and CPU, which can lead
   * to overhead in the reported results.
   *
   * Example usage:
   * \code{.cpp}
   * Timer t = Timer::Start(Device::cpu());
   * my_long_running_function();
   * t->Stop();
   * ... // some more computation
   * int64_t nanosecs = t->SyncAndGetElapsedNanos() // elapsed time in nanoseconds
   * \endcode
   *
   * To add a new device-specific timer, register a new function
   * "profiler.timer.my_device" (where `my_device` is the `DeviceName` of your
   * device). This function should accept a `Device` and return a new `Timer`
   * that has already been started.
   *
   * For example, this is how the CPU timer is implemented:
   * \code{.cpp}
   *  class CPUTimerNode : public TimerNode {
   *   public:
   *    virtual void Start() { start_ = std::chrono::high_resolution_clock::now(); }
   *    virtual void Stop() { duration_ = std::chrono::high_resolution_clock::now() - start_; }
   *    virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
   *    virtual ~CPUTimerNode() {}
   *
   *    static constexpr const char* _type_key = "CPUTimerNode";
   *    TVM_DECLARE_FINAL_OBJECT_INFO(CPUTimerNode, TimerNode);
   *
   *   private:
   *    std::chrono::high_resolution_clock::time_point start_;
   *    std::chrono::duration<int64_t, std::nano> duration_;
   *  };
   *  TVM_REGISTER_OBJECT_TYPE(CPUTimerNode);
   *
   *  TVM_REGISTER_GLOBAL("profiling.timer.cpu").set_body_typed([](Device dev) {
   *    return Timer(make_object<CPUTimerNode>());
   *  });
   * \endcode
   */
  static TVM_DLL Timer Start(Device dev);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Timer, ObjectRef, TimerNode);
};

/*!
 * \brief Default timer if one does not exist for the device.
 * \param dev The device to time on.
 *
 * Note that this timer performs synchronization between the device and CPU,
 * which can lead to overhead in the reported results.
 */
Timer DefaultTimer(Device dev);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_PROFILING_H_
