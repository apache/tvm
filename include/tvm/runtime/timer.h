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
 * \file include/tvm/runtime/timer.h
 * \brief Runtime timer primitives: Timer, TimerNode, WrapTimeEvaluator.
 */
#ifndef TVM_RUNTIME_TIMER_H_
#define TVM_RUNTIME_TIMER_H_

#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

namespace tvm {
namespace runtime {

/*! \brief Base class for all timer implementations.
 *
 * New implementations of this interface should make sure that `Start` and `Stop`
 * are as lightweight as possible. Expensive state synchronization should be
 * done in `SyncAndGetElapsedNanos`.
 */
class TimerNode : public ffi::Object {
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

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("runtime.TimerNode", TimerNode, ffi::Object);
};

/*! \brief Timer for a specific device.
 *
 * This is a managed reference to a TimerNode.
 *
 * \sa TimerNode
 */
class Timer : public ffi::ObjectRef {
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
   * "runtime.timer.my_device" (where `my_device` is the `DeviceName` of your
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
   *    static constexpr const char* _type_key = "runtime.CPUTimerNode";
   *    TVM_FFI_DECLARE_OBJECT_INFO_FINAL(CPUTimerNode, TimerNode);
   *
   *   private:
   *    std::chrono::high_resolution_clock::time_point start_;
   *    std::chrono::duration<int64_t, std::nano> duration_;
   *  };
   *
   *
   *  TVM_FFI_STATIC_INIT_BLOCK() {
   *    namespace refl = tvm::ffi::reflection;
   *    refl::GlobalDef().def("runtime.timer.cpu", [](Device dev) {
   *      return Timer(ffi::make_object<CPUTimerNode>());
   *    });
   *  }
   * \endcode
   */
  static TVM_RUNTIME_DLL Timer Start(Device dev);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Timer, ffi::ObjectRef, TimerNode);
};

/*!
 * \brief Wrap a timer function to measure the time cost of a given packed function.
 *
 * Approximate implementation:
 * \code{.py}
 * f() // warmup
 * for i in range(repeat)
 *   f_preproc()
 *   while True:
 *     start = time()
 *     for j in range(number):
 *       f()
 *     duration_ms = time() - start
 *     if duration_ms >= min_repeat_ms:
 *       break
 *     else:
 *        number = (min_repeat_ms / (duration_ms / number) + 1
 *   if cooldown_interval_ms and i % repeats_to_cooldown == 0:
 *     sleep(cooldown_interval_ms)
 * \endcode
 *
 * \param f The function argument.
 * \param dev The device.
 * \param number The number of times to run this function for taking average.
 *        We call these runs as one `repeat` of measurement.
 * \param repeat The number of times to repeat the measurement.
 *        In total, the function will be invoked (1 + number x repeat) times,
 *        where the first one is warm up and will be discarded.
 *        The returned result contains `repeat` costs,
 *        each of which is an average of `number` costs.
 * \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
 *        By default, one `repeat` contains `number` runs. If this parameter is set,
 *        the parameters `number` will be dynamically adjusted to meet the
 *        minimum duration requirement of one `repeat`.
 *        i.e., When the run time of one `repeat` falls below this time,
 *        the `number` parameter will be automatically increased.
 * \param limit_zero_time_iterations The maximum number of repeats when
 *        measured time is equal to 0.  It helps to avoid hanging during measurements.
 * \param cooldown_interval_ms The cooldown interval in milliseconds between the number of repeats
 *        defined by `repeats_to_cooldown`.
 * \param repeats_to_cooldown The number of repeats before the
 *        cooldown is activated.
 * \param cache_flush_bytes The number of bytes to flush from cache before
 * \param f_preproc The function to be executed before we execute time
 *        evaluator.
 * \return f_timer A timer function.
 */
ffi::Function WrapTimeEvaluator(ffi::Function f, Device dev, int number, int repeat,
                                int min_repeat_ms, int limit_zero_time_iterations,
                                int cooldown_interval_ms, int repeats_to_cooldown,
                                int cache_flush_bytes = 0, ffi::Function f_preproc = nullptr);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TIMER_H_
