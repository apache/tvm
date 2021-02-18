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

/*!
 * \brief Default timer if one does not exist to the platform.
 * \param ctx The context to time on.
 *
 * Note that this timer performs synchronization between the device and CPU,
 * which can lead to overhead in the reported results.
 */
TypedPackedFunc<int64_t()> DefaultTimer(TVMContext ctx);

/*!
 * \brief Get a device specific timer.
 * \param ctx The device context to time.
 * \return A function, that when called starts a timer. The results from this
 *         function is another function that will stop the timer and return the elapsed
 *         time in nanoseconds.
 *
 * Users can register a timer for a device by registering a packed function
 * with the name "profiler.timer.device_name". This function should take a
 * TVMContext and return a new function. The new function should return the
 * elapsed time between the first and second call in nanoseconds.
 *
 * Note that timers are specific to a context (and by extension device stream).
 * The code being timed should run on the specific context only, otherwise you
 * may get mixed results. Furthermore, the context should not be modified
 * between the start and end of the timer (i.e. do not call TVMDeviceSetStream).
 *
 * Example usage:
 * \code{.cpp}
 * auto timer_stop = StartTimer(TVMContext::cpu());
 * my_long_running_function();
 * int64_t nanosecs = timer_stop(); // elapsed time in nanoseconds
 * \endcode
 */
inline TypedPackedFunc<int64_t()> StartTimer(TVMContext ctx) {
  auto f = Registry::Get(std::string("profiling.timer.") + DeviceName(ctx.device_type));
  if (f == nullptr) {
    return DefaultTimer(ctx);
  } else {
    return f->operator()(ctx);
  }
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_PROFILING_H_
