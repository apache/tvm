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
 * \file src/runtime/timer.cc
 * \brief Runtime timer primitives: Timer, WrapTimeEvaluator.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/timer.h>

#include <chrono>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace tvm {
namespace runtime {

class DefaultTimerNode : public TimerNode {
 public:
  virtual void Start() {
    DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    start_ = std::chrono::high_resolution_clock::now();
  }
  virtual void Stop() {
    DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    duration_ = std::chrono::high_resolution_clock::now() - start_;
  }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~DefaultTimerNode() {}

  explicit DefaultTimerNode(Device dev) : device_(dev) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.DefaultTimerNode", DefaultTimerNode, TimerNode);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
  Device device_;
};

static Timer DefaultTimer(Device dev) { return Timer(ffi::make_object<DefaultTimerNode>(dev)); }

class CPUTimerNode : public TimerNode {
 public:
  virtual void Start() { start_ = std::chrono::high_resolution_clock::now(); }
  virtual void Stop() { duration_ = std::chrono::high_resolution_clock::now() - start_; }
  virtual int64_t SyncAndGetElapsedNanos() { return duration_.count(); }
  virtual ~CPUTimerNode() {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.CPUTimerNode", CPUTimerNode, TimerNode);

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::duration<int64_t, std::nano> duration_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.cpu",
                        [](Device dev) { return Timer(ffi::make_object<CPUTimerNode>()); });
}

namespace {
// keep track of which timers are not defined but we have already warned about
std::set<DLDeviceType> seen_devices;
std::mutex seen_devices_lock;
}  // namespace

Timer Timer::Start(Device dev) {
  auto f = tvm::ffi::Function::GetGlobal(std::string("runtime.timer.") +
                                         DLDeviceType2Str(dev.device_type));
  if (!f.has_value()) {
    {
      std::lock_guard<std::mutex> lock(seen_devices_lock);
      if (seen_devices.find(dev.device_type) == seen_devices.end()) {
        LOG(WARNING)
            << "No timer implementation for " << DLDeviceType2Str(dev.device_type)
            << ", using default timer instead. It may be inaccurate or have extra overhead.";
        seen_devices.insert(dev.device_type);
      }
    }
    Timer t = DefaultTimer(dev);
    t->Start();
    return t;
  } else {
    Timer t = f->operator()(dev).cast<Timer>();
    t->Start();
    return t;
  }
}

ffi::Function WrapTimeEvaluator(ffi::Function pf, Device dev, int number, int repeat,
                                int min_repeat_ms, int limit_zero_time_iterations,
                                int cooldown_interval_ms, int repeats_to_cooldown,
                                int cache_flush_bytes, ffi::Function f_preproc) {
  TVM_FFI_ICHECK(pf != nullptr);

  auto ftimer = [pf, dev, number, repeat, min_repeat_ms, limit_zero_time_iterations,
                 cooldown_interval_ms, repeats_to_cooldown, cache_flush_bytes,
                 f_preproc](const ffi::AnyView* args, int num_args, ffi::Any* rv) mutable {
    ffi::Any temp;
    std::ostringstream os;
    // skip first time call, to activate lazy compilation components.
    pf.CallPacked(args, num_args, &temp);

    // allocate two large arrays to flush L2 cache
    Tensor arr1, arr2;
    if (cache_flush_bytes > 0) {
      arr1 = Tensor::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
      arr2 = Tensor::Empty({cache_flush_bytes / 4}, {kDLInt, 32, 1}, dev);
    }

    DeviceAPI::Get(dev)->StreamSync(dev, nullptr);

    for (int i = 0; i < repeat; ++i) {
      if (f_preproc != nullptr) {
        f_preproc.CallPacked(args, num_args, &temp);
      }
      double duration_ms = 0.0;
      int absolute_zero_times = 0;
      do {
        if (duration_ms > 0.0) {
          const double golden_ratio = 1.618;
          number = static_cast<int>(
              std::max((min_repeat_ms / (duration_ms / number) + 1), number * golden_ratio));
        }
        if (cache_flush_bytes > 0) {
          arr1.CopyFrom(arr2);
        }
        DeviceAPI::Get(dev)->StreamSync(dev, nullptr);
        // start timing
        Timer t = Timer::Start(dev);
        for (int j = 0; j < number; ++j) {
          pf.CallPacked(args, num_args, &temp);
        }
        t->Stop();
        int64_t t_nanos = t->SyncAndGetElapsedNanos();
        if (t_nanos == 0) absolute_zero_times++;
        duration_ms = t_nanos / 1e6;
      } while (duration_ms < min_repeat_ms && absolute_zero_times < limit_zero_time_iterations);

      double speed = duration_ms / 1e3 / number;
      os.write(reinterpret_cast<char*>(&speed), sizeof(speed));

      if (cooldown_interval_ms > 0 && (i % repeats_to_cooldown) == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cooldown_interval_ms));
      }
    }

    std::string blob = os.str();
    // return the time.
    *rv = ffi::Bytes(std::move(blob));
  };
  return ffi::Function::FromPacked(ftimer);
}

}  // namespace runtime
}  // namespace tvm
