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
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

namespace profiling {
/*! \brief Wrapper for `Device` because `Device` is not passable across the
 * PackedFunc interface.
 */
struct DeviceWrapperNode : public Object {
  /*! The device */
  Device device;

  /*! Constructor */
  explicit DeviceWrapperNode(Device device) : device(device) {}

  static constexpr const char* _type_key = "runtime.profiling.DeviceWrapper";
  TVM_DECLARE_BASE_OBJECT_INFO(DeviceWrapperNode, Object);
};

/*! \brief Wrapper for `Device`. */
class DeviceWrapper : public ObjectRef {
 public:
  explicit DeviceWrapper(Device dev) { data_ = make_object<DeviceWrapperNode>(dev); }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(DeviceWrapper, ObjectRef, DeviceWrapperNode);
};

/*! \brief Data collected from a profiling run. Includes per-call metrics and per-device metrics.
 */
class ReportNode : public Object {
 public:
  /*! \brief A list of function calls and the metrics recorded for that call.
   *
   * Each element is a mapping from metric name to value. Some metrics that
   * appear in every call are "Name" (the function name), "Argument Shapes",
   * and "Duration (us)". Values are one of `String`, `PercentNode`,
   * `DurationNode`, or `CountNode`.
   */
  Array<Map<String, ObjectRef>> calls;
  /*! \brief Metrics collected for the entire run of the model on a per-device basis.
   *
   * `device_metrics` is indexed by device name then metric.
   *
   * These metrics may be larger than the sum of the same metric in `calls`
   * because these metrics include the overhead of the executor.
   */
  Map<String, Map<String, ObjectRef>> device_metrics;
  /*! Configuration used for this profiling run. Includes number of threads, executor.
   *
   * Values must be an object type that can be used with device_metrics.
   */
  Map<String, ObjectRef> configuration;
  /*! \brief Output `calls` in CSV format.
   *
   * Note that this does not include `device_metrics`, it only includes per-call metrics.
   */
  String AsCSV() const;
  /*! \brief Create a human readable table of profiling metrics.
   *
   *  \param aggregate Whether or not to join multiple calls to the
   *      same op into a single line.
   *
   *  \param sort Whether or not to sort call frames by descending
   *      duration. If false and if `aggregate` is false, frames will
   *      be sorted by order of appearance in the program. Order is
   *      undefined if `sort` is false and `aggregate` is true.
   *
   *  \param compute_col_sums Whether or not to include sum totals for
   *      the Count, Duation, and Percent columns.
   *
   */
  String AsTable(bool sort = true, bool aggregate = true, bool compute_col_sums = true) const;
  /*! \brief Convert this report to JSON.
   *
   * Output JSON will be of this format:
   * \code
   *  {
   *    "calls": [
   *      {
   *        "Duration (us)": {
   *          "microseconds": 12.3
   *        },
   *        "Name": "fused_dense",
   *        "Count": {
   *          "count": 1
   *        },
   *        "Percent": {
   *          "percent": 10.3
   *        }
   *      }
   *    ],
   *    "device_metrics": {
   *      "cpu": {
   *        "Duration (us)": {
   *          "microseconds": 334.2
   *        },
   *        "Percent": {
   *          "percent": 100
   *        }
   *      }
   *    }
   *  }
   * \endcode
   */
  String AsJSON() const;

  static constexpr const char* _type_key = "runtime.profiling.Report";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReportNode, Object);
};

class Report : public ObjectRef {
 public:
  /*! Construct a Report from a set of calls (with associated metrics) and per-device metrics.
   * \param calls Function calls and associated metrics.
   * \param device_metrics Per-device metrics for overall execution.
   * \param configuration Configuration data specific to this profiling run.
   */
  explicit Report(Array<Map<String, ObjectRef>> calls,
                  Map<String, Map<String, ObjectRef>> device_metrics,
                  Map<String, ObjectRef> configuration);

  /*! Deserialize a Report from a JSON object. Needed for sending the report over RPC.
   * \param json Serialized json report from `ReportNode::AsJSON`.
   * \returns A Report.
   */
  static Report FromJSON(String json);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Report, ObjectRef, ReportNode);
};

/*! \brief Interface for user defined profiling metric collection.
 *
 * Users can register their own collector by registering a packed function with
 * the name "runtime.profiling.metrics.my_collector_name" where
 * "my_collector_name" is the name of their collector. This function should
 * take an Array of Device as input which contains the devices the collector
 * will be run on.
 *
 * `MetricCollectorNode`s will be called in the following fashion.
 * \code
 * MetricCollector mc;
 * for (auto op : model) {
 *   auto o = mc.Start();
 *   op();
 *   auto metrics = mc.Stop(o); // metrics are added the profiling report
 * }
 * \endcode
 */
class MetricCollectorNode : public Object {
 public:
  /*! \brief Initialization call. Called before profiling has started. Any
   * expensive precomputation should happen here.
   * \param devs The list of devices this collector will be run on.
   */
  virtual void Init(Array<DeviceWrapper> devs) = 0;
  /*! \brief Start colling metrics for a function call.
   * \param dev The device the call will be run on.
   * \returns An object used to maintain state of the metric collection. This
   * object will be passed to the corresponding `Stop` call. If the device is
   * not supported, this function will return a nullptr ObjectRef.
   */
  virtual ObjectRef Start(Device dev) = 0;
  /*! \brief Stop collecting metrics.
   * \param obj The object created by the corresponding `Start` call.
   * \returns A set of metric names and the associated values. Values must be
   * one of DurationNode, PercentNode, CountNode, or StringObj.
   */
  virtual Map<String, ObjectRef> Stop(ObjectRef obj) = 0;

  virtual ~MetricCollectorNode() {}

  static constexpr const char* _type_key = "runtime.profiling.MetricCollector";
  TVM_DECLARE_BASE_OBJECT_INFO(MetricCollectorNode, Object);
};

/*! \brief Wrapper for `MetricCollectorNode`. */
class MetricCollector : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MetricCollector, ObjectRef, MetricCollectorNode);
};

/*! Information about a single function or operator call. */
struct CallFrame {
  /*! Device on which the call was made */
  Device dev;
  /*! Name of the function or op */
  String name;
  /*! Runtime of the function or op */
  Timer timer;
  /*! Extra performance metrics */
  std::unordered_map<std::string, ObjectRef> extra_metrics;
  /*! User defined metric collectors. Each pair is the MetricCollector and its
   * associated data (returned from MetricCollector.Start).
   */
  std::vector<std::pair<MetricCollector, ObjectRef>> extra_collectors;
};

/*! Runtime profiler for function and/or operator calls. Used in the graph
 * runtime and VM to provide profiling information for all operators.
 *
 * Example usage:
 * \code{.cpp}
 * Device cpu, gpu;
 * Profiler prof({cpu, gpu});
 * my_gpu_kernel(); // do a warmup iteration
 * prof.Start();
 * prof.StartCall("my_gpu_kernel", gpu);
 * my_gpu_kernel();
 * prof.StopCall();
 * prof.StartCall("my_cpu_function", cpu);
 * my_cpu_function();
 * prof.StopCall();
 * prof.Stop();
 * std::cout << prof.Report << std::endl; // print profiling report
 * \endcode
 */
class Profiler {
 public:
  /*! Constructor.
   *
   * The profiler should be constructed before you do any warmup iterations.
   *
   * \note
   * Calling this constructor will reset the TVM threadpool. It is necessary in
   * order to install thread handlers required by certain collectors.
   *
   * \param devs The list of devices the profiler will be running on. Should
   *             include all devices used by profiled operators.
   * \param metric_collectors Additional `MetricCollector`s to use with this profiler.
   * \param configuration Additional configuration data to add to the outputted profiling report.
   */
  explicit Profiler(std::vector<Device> devs, std::vector<MetricCollector> metric_collectors,
                    std::unordered_map<String, ObjectRef> configuration = {});
  /*! \brief Start the profiler.
   *
   * This function should only be called once per object.
   */
  void Start();
  /*! \brief Stop the profiler.
   *
   * This function should only be called once per object after start has been called.
   */
  void Stop();
  /*! \brief Start a function call.
   * \param name The name of the function being called.
   * \param dev The device on which the function is running.
   * \param extra_metrics Optional additional profiling information to add to
   * the frame (input sizes, allocations).
   *
   * `StartCall` may be nested, but each `StartCall` needs a matching
   * `StopCall`. Function calls are stopped in LIFO order, so calls to
   * `StartCall` and `StopCall` must be nested properly.
   */
  void StartCall(String name, Device dev,
                 std::unordered_map<std::string, ObjectRef> extra_metrics = {});
  /*! \brief Stop the last `StartCall`.
   * \param extra_metrics Optional additional profiling information to add to
   * the frame (input sizes, allocations).
   */
  void StopCall(std::unordered_map<std::string, ObjectRef> extra_metrics = {});
  /*! \brief A report of total runtime between `Start` and `Stop` as
   *        well as individual statistics for each `StartCall`-`StopCall` pair.
   *  \returns A `Report` that can either be formatted as CSV (with `.AsCSV`)
   *  or as a human readable table (with `.AsTable`).
   */
  profiling::Report Report();
  /*! \brief Check if the profiler is currently running.
   * \returns Whether or not the profiler is running.
   */
  bool IsRunning() const { return is_running_; }

 private:
  std::vector<Device> devs_;
  bool is_running_{false};
  std::vector<CallFrame> calls_;
  std::stack<CallFrame> in_flight_;
  std::vector<MetricCollector> collectors_;
  std::unordered_map<String, ObjectRef> configuration_;
};

/* \brief A duration in time. */
class DurationNode : public Object {
 public:
  /* The duration as a floating point number of microseconds. */
  double microseconds;

  /* \brief Construct a new duration.
   * \param a The duration in microseconds.
   */
  explicit DurationNode(double a) : microseconds(a) {}

  static constexpr const char* _type_key = "runtime.profiling.Duration";
  TVM_DECLARE_FINAL_OBJECT_INFO(DurationNode, Object);
};

/* A percentage of something */
class PercentNode : public Object {
 public:
  /* The percent as a floating point value out of 100%. i.e. if `percent` is 10 then we have 10%. */
  double percent;

  /* \brief Construct a new percentage.
   * \param a The percentage out of 100.
   */
  explicit PercentNode(double a) : percent(a) {}

  static constexpr const char* _type_key = "runtime.profiling.Percent";
  TVM_DECLARE_FINAL_OBJECT_INFO(PercentNode, Object);
};

/* A count of something */
class CountNode : public Object {
 public:
  /* The actual count */
  int64_t value;

  /* \brief Construct a new count.
   * \param a The count.
   */
  explicit CountNode(int64_t a) : value(a) {}

  static constexpr const char* _type_key = "runtime.profiling.Count";
  TVM_DECLARE_FINAL_OBJECT_INFO(CountNode, Object);
};

/* \brief A ratio of two things. */
class RatioNode : public Object {
 public:
  /* The ratio as a double precision floating point number. */
  double ratio;

  /* \brief Construct a new ratio.
   * \param a The ratio.
   */
  explicit RatioNode(double a) : ratio(a) {}

  static constexpr const char* _type_key = "runtime.profiling.Ratio";
  TVM_DECLARE_FINAL_OBJECT_INFO(RatioNode, Object);
};

/*! \brief String representation of an array of NDArray shapes
 *  \param shapes Array of NDArrays to get the shapes of.
 *  \return A textual representation of the shapes. For example: `float32[2], int64[1, 2]`.
 */
String ShapeString(const std::vector<NDArray>& shapes);
/*! \brief String representation of shape encoded as an NDArray
 *  \param shape NDArray containing the shape.
 *  \param dtype The dtype of the shape.
 *  \return A textual representation of the shape. For example: `float32[2]`.
 */
String ShapeString(NDArray shape, DLDataType dtype);
/*! \brief String representation of a shape encoded as a vector
 *  \param shape Shape as a vector of integers.
 *  \param dtype The dtype of the shape.
 *  \return A textual representation of the shape. For example: `float32[2]`.
 */
String ShapeString(const std::vector<int64_t>& shape, DLDataType dtype);

/*! \brief Collect performance information of a function execution. Usually
 * used with a compiled PrimFunc (via tvm.build).
 *
 * This information can include performance counters like cache hits and FLOPs
 * that are useful in debugging performance issues of individual PrimFuncs.
 * Different metrics can be collected depending on which MetricCollector is
 * used.
 *
 * Example usage:
 * \code{.cpp}
 * // Use PAPI to measure the number of floating point operations.
 * PackedFunc profiler = ProfileModule(
 *     mod, "main", kDLCPU, 0, {CreatePAPIMetricCollector({{kDLCPU, 0}, {"PAPI_FP_OPS"}})});
 * Report r = profiler(arg1, arg2, arg);
 * std::cout << r << std::endl;
 * \endcode
 *
 * \param mod Module to profile. Usually a PrimFunc that has been compiled to machine code.
 * \param func_name Name of function to run in the module.
 * \param device_type Device type to run on. Profiling will include performance
 *                    metrics specific to this device type.
 * \param device_id Id of device to run on.
 * \param warmup_iters Number of iterations of the function to run before collecting
 *                     performance information. Recommend to set this larger
 *                     than 0 so that cache effects are consistent.
 * \param collectors List of different
 *                   ways to collect metrics. See MetricCollector.
 * \returns A PackedFunc which takes the same arguments as the `mod[func_name]`
 *          and returns performance metrics as a `Map<String, ObjectRef>` where
 *          values can be `CountNode`, `DurationNode`, `PercentNode`.
 */
PackedFunc ProfileFunction(Module mod, std::string func_name, int device_type, int device_id,
                           int warmup_iters, Array<MetricCollector> collectors);

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
PackedFunc WrapTimeEvaluator(PackedFunc f, Device dev, int number, int repeat, int min_repeat_ms,
                             int limit_zero_time_iterations, int cooldown_interval_ms,
                             int repeats_to_cooldown, int cache_flush_bytes = 0,
                             PackedFunc f_preproc = nullptr);

}  // namespace profiling
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_PROFILING_H_
