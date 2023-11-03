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
#include <papi.h>
#include <tvm/runtime/contrib/papi.h>

#include <string>
#include <vector>

namespace tvm {
namespace runtime {
namespace profiling {

#define PAPI_CALL(func)                                             \
  {                                                                 \
    int e = (func);                                                 \
    if (e != PAPI_OK) {                                             \
      LOG(FATAL) << "PAPIError: in function " #func " " << e << " " \
                 << std::string(PAPI_strerror(e));                  \
    }                                                               \
  }

static const std::unordered_map<DLDeviceType, std::vector<std::string>> default_metric_names = {
    {kDLCPU,
     {"perf::CYCLES", "perf::STALLED-CYCLES-FRONTEND", "perf::STALLED-CYCLES-BACKEND",
      "perf::INSTRUCTIONS", "perf::CACHE-MISSES"}},
    {kDLCUDA, {"cuda:::event:elapsed_cycles_sm:device=0"}}};

/*! \brief Object that holds the values of counters at the start of a function call. */
struct PAPIEventSetNode : public Object {
  /*! \brief The starting values of counters for all metrics of a specific device. */
  std::vector<long_long> start_values;
  /*! \brief The device these counters are for. */
  Device dev;

  explicit PAPIEventSetNode(std::vector<long_long> start_values, Device dev)
      : start_values(start_values), dev(dev) {}

  static constexpr const char* _type_key = "PAPIEventSetNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(PAPIEventSetNode, Object);
};

/* Get the PAPI component id for the given device.
 * \param dev The device to get the component for.
 * \returns PAPI component id for the device. Returns -1 if the device is not
 * supported by PAPI.
 */
int component_for_device(Device dev) {
  std::string component_name;
  switch (dev.device_type) {
    case kDLCPU:
      component_name = "perf_event";
      break;
    case kDLCUDA:
      component_name = "cuda";
      break;
    case kDLROCM:
      component_name = "rocm";
      break;
    default:
      LOG(WARNING) << "PAPI does not support device " << DLDeviceType2Str(dev.device_type);
      return -1;
  }
  int cidx = PAPI_get_component_index(component_name.c_str());
  if (cidx < 0) {
    LOG(FATAL) << "Cannot find PAPI component \"" << component_name
               << "\". Maybe you need to build PAPI with support for this component (use "
                  "`./configure --with-components="
               << component_name << "`).";
  }
  return cidx;
}

/*! \brief MetricCollectorNode for PAPI metrics.
 *
 * PAPI (Performance Application Programming Interface) collects metrics on a
 * variety of platforms including cpu, cuda and rocm.
 *
 * PAPI is avaliable at https://bitbucket.org/icl/papi/src/master/.
 */
struct PAPIMetricCollectorNode final : public MetricCollectorNode {
  /*! \brief Construct a metric collector that collects a specific set of metrics.
   *
   * \param metrics A mapping from a device type to the metrics that should be
   * collected on that device. You can find the names of available metrics by
   * running `papi_native_avail`.
   */
  explicit PAPIMetricCollectorNode(Map<DeviceWrapper, Array<String>> metrics) {
    for (auto& p : metrics) {
      papi_metric_names[p.first->device] = {};
      for (auto& metric : p.second) {
        papi_metric_names[p.first->device].push_back(metric);
      }
    }
  }
  explicit PAPIMetricCollectorNode() {}

  /*! \brief Initialization call.
   * \param devices The devices this collector will be running on
   */
  void Init(Array<DeviceWrapper> devices) {
    if (!PAPI_is_initialized()) {
      if (sizeof(long_long) > sizeof(int64_t)) {
        LOG(WARNING) << "PAPI's long_long is larger than int64_t. Overflow may occur when "
                        "reporting metrics.";
      }
      CHECK_EQ(PAPI_library_init(PAPI_VER_CURRENT), PAPI_VER_CURRENT)
          << "Error while initializing PAPI";
    }

    // If no metrics were provided we use the default set. The names were not
    // initialized in the constructor because we did not know which devices we
    // were running on.
    if (papi_metric_names.size() == 0) {
      for (auto wrapped_device : devices) {
        Device device = wrapped_device->device;
        auto it = default_metric_names.find(device.device_type);
        if (it != default_metric_names.end()) {
          papi_metric_names[device] = it->second;
        }
      }
    }

    // create event sets for each device
    for (auto wrapped_device : devices) {
      Device device = wrapped_device->device;
      int cidx = component_for_device(device);
      // unknown device, skipping
      if (cidx < 0) {
        continue;
      }

      auto it = papi_metric_names.find(device);
      // skip devices with no metrics defined
      if (it == papi_metric_names.end() || it->second.size() == 0) {
        continue;
      }
      auto& metric_names = it->second;

      const PAPI_component_info_t* component = PAPI_get_component_info(cidx);
      if (component->disabled) {
        std::string help_message = "";
        switch (device.device_type) {
          case kDLCPU:
            help_message =
                "Try setting `sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'`";
            break;
          case kDLCUDA:
            help_message =
                "Try enabling gpu profiling with `modprobe nvidia "
                "NVreg_RestrictProfilingToAdminUsers=0`. If that does not work, try adding  "
                "`options nvidia \"NVreg_RestrictProfilingToAdminUsers=0\"` to "
                "`/etc/modprobe.d/nvidia-kernel-common.conf`.";
            break;
          default:
            break;
        }
        LOG(WARNING) << "PAPI could not initialize counters for "
                     << DLDeviceType2Str(device.device_type) << ": " << component->disabled_reason
                     << "\n"
                     << help_message;
        continue;
      }

      int event_set = PAPI_NULL;
      PAPI_CALL(PAPI_create_eventset(&event_set));
      PAPI_CALL(PAPI_assign_eventset_component(event_set, cidx));
      if (device.device_type == kDLCPU) {
        // we set PAPI_INHERIT to make it so threads created after this inherit the event_set.
        PAPI_option_t opt;
        memset(&opt, 0x0, sizeof(PAPI_option_t));
        opt.inherit.inherit = PAPI_INHERIT_ALL;
        opt.inherit.eventset = event_set;
        PAPI_CALL(PAPI_set_opt(PAPI_INHERIT, &opt));
      }

      if (static_cast<int>(metric_names.size()) > PAPI_num_cmp_hwctrs(cidx)) {
        PAPI_CALL(PAPI_set_multiplex(event_set));
      }

      // add all the metrics
      for (auto metric : metric_names) {
        int e = PAPI_add_named_event(event_set, metric.c_str());
        if (e != PAPI_OK) {
          LOG(FATAL) << "PAPIError: " << e << " " << std::string(PAPI_strerror(e)) << ": " << metric
                     << ".";
        }
      }
      // Because we may have multiple calls in flight at the same time, we
      // start all the timers when we initialize. Then we calculate the metrics
      // counts for a call by comparing counter values at the start vs end of
      // the call.
      PAPI_CALL(PAPI_start(event_set));
      event_sets[device] = event_set;
    }
  }
  /*! \brief Called right before a function call. Reads starting values of the
   * measured metrics.
   *
   * \param dev The device the function will be run on.
   * \returns A `PAPIEventSetNode` containing values for the counters at the
   * start of the call. Passed to a corresponding `Stop` call.
   */
  ObjectRef Start(Device dev) final {
    // Record counter values at the start of the call, so we can calculate the
    // metrics for the call by comparing the values at the end of the call.
    auto it = event_sets.find(dev);
    if (it != event_sets.end()) {
      int event_set = it->second;
      std::vector<long_long> values(papi_metric_names[dev].size());
      PAPI_CALL(PAPI_read(event_set, values.data()));
      return ObjectRef(make_object<PAPIEventSetNode>(values, dev));
    } else {
      return ObjectRef(nullptr);
    }
  }
  /*! \brief Called right after a function call. Reads ending values of the
   * measured metrics. Computes the change in each metric from the
   * corresponding `Start` call.
   *
   * \param obj `PAPIEventSetNode` created by a call to `Start`.
   * \returns A mapping from metric name to value.
   */
  Map<String, ObjectRef> Stop(ObjectRef obj) final {
    const PAPIEventSetNode* event_set_node = obj.as<PAPIEventSetNode>();
    std::vector<long_long> end_values(papi_metric_names[event_set_node->dev].size());
    PAPI_CALL(PAPI_read(event_sets[event_set_node->dev], end_values.data()));
    std::unordered_map<String, ObjectRef> reported_metrics;
    for (size_t i = 0; i < end_values.size(); i++) {
      if (end_values[i] < event_set_node->start_values[i]) {
        LOG(WARNING) << "Detected overflow when reading performance counter, setting value to -1.";
        reported_metrics[papi_metric_names[event_set_node->dev][i]] =
            ObjectRef(make_object<CountNode>(-1));
      } else {
        reported_metrics[papi_metric_names[event_set_node->dev][i]] =
            ObjectRef(make_object<CountNode>(end_values[i] - event_set_node->start_values[i]));
      }
    }
    return reported_metrics;
  }

  ~PAPIMetricCollectorNode() final {
    for (auto p : event_sets) {
      PAPI_CALL(PAPI_stop(p.second, NULL));
      PAPI_CALL(PAPI_cleanup_eventset(p.second));
      PAPI_CALL(PAPI_destroy_eventset(&p.second));
    }
  }

  /*! \brief Device-specific event sets. Contains the running counters (the int values) for that
   * device. */
  std::unordered_map<Device, int> event_sets;
  /*! \brief Device-specific metric names. Order of names matches the order in the corresponding
   * `event_set`. */
  std::unordered_map<Device, std::vector<std::string>> papi_metric_names;

  static constexpr const char* _type_key = "runtime.profiling.PAPIMetricCollector";
  TVM_DECLARE_FINAL_OBJECT_INFO(PAPIMetricCollectorNode, MetricCollectorNode);
};

/*! \brief Wrapper for `PAPIMetricCollectorNode`. */
class PAPIMetricCollector : public MetricCollector {
 public:
  explicit PAPIMetricCollector(Map<DeviceWrapper, Array<String>> metrics) {
    data_ = make_object<PAPIMetricCollectorNode>(metrics);
  }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PAPIMetricCollector, MetricCollector,
                                        PAPIMetricCollectorNode);
};

MetricCollector CreatePAPIMetricCollector(Map<DeviceWrapper, Array<String>> metrics) {
  return PAPIMetricCollector(metrics);
}

TVM_REGISTER_OBJECT_TYPE(PAPIEventSetNode);
TVM_REGISTER_OBJECT_TYPE(PAPIMetricCollectorNode);

TVM_REGISTER_GLOBAL("runtime.profiling.PAPIMetricCollector")
    .set_body_typed([](Map<DeviceWrapper, Array<String>> metrics) {
      return PAPIMetricCollector(metrics);
    });

}  // namespace profiling
}  // namespace runtime
}  // namespace tvm
