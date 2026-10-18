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
 * \file tvm/runtime/timer.h
 * \brief Device timer interface and object reference.
 */
#ifndef TVM_RUNTIME_TIMER_H_
#define TVM_RUNTIME_TIMER_H_

#include <tvm/ffi/object.h>

namespace tvm {
namespace runtime {

/*! \brief Base class for all timer implementations.
 *
 * New implementations of this interface should make sure that `Start` and `Stop`
 * are as lightweight as possible. Expensive state synchronization should be
 * done in `SyncAndGetElapsedNanos`.
 */
class TimerObj : public ffi::Object {
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

  virtual ~TimerObj() {}

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("runtime.Timer", TimerObj, ffi::Object);
};

/*! \brief Timer for a specific device.
 *
 * This is a managed reference to a TimerObj.
 *
 * \sa TimerObj
 */
class Timer : public ffi::ObjectRef {
 public:
  explicit Timer(ffi::UnsafeInit tag, ffi::ObjectPtr<TimerObj> n) : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Timer, ffi::ObjectRef, TimerObj);
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TIMER_H_
