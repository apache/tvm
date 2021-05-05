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
 * \brief Performance counters for profiling via the PAPI library.
 */
#ifndef TVM_RUNTIME_CONTRIB_PAPI_PAPI_H_
#define TVM_RUNTIME_CONTRIB_PAPI_PAPI_H_

#include <papi.h>
#include <tvm/runtime/profiling.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace profiling {

#define PAPI_CALL(func)                                                         \
  {                                                                             \
    int e = (func);                                                             \
    if (e < 0) {                                                                \
      LOG(FATAL) << "PAPIError: " << e << " " << std::string(PAPI_strerror(e)); \
    }                                                                           \
  }

static const std::unordered_map<DLDeviceType, std::vector<std::string>> default_metrics = {
    {kDLCPU,
     {"perf::CYCLES", "perf::STALLED-CYCLES-FRONTEND", "perf::STALLED-CYCLES-BACKEND",
      "perf::INSTRUCTIONS", "perf::CACHE-MISSES"}},
    {kDLGPU, {"cuda:::event:elapsed_cycles_sm:device=0"}}};

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

int component_for_device(Device dev) {
  std::string component_name;
  switch (dev.device_type) {
    case kDLCPU:
    case kDLCPUPinned:
      component_name = "perf_event";
      break;
    case kDLGPU:
      component_name = "cuda";
      break;
    case kDLROCM:
      component_name = "rocm";
      break;
    default:
      LOG(WARNING) << "PAPI does not support device " << DeviceName(dev.device_type);
      return -1;
  }
  int cidx = PAPI_get_component_index(component_name.c_str());
  if (cidx < 0) {
    LOG(FATAL) << "Cannot find PAPI component \"" << component_name
               << "\". Maybe you need to build PAPI with support for this component (use "
                  "`./configure --components="
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
 *
 * Users can change the metrics collected for by setting the environment
 * variable `TVM_PAPI_${device_name}_METRICS` with a semicolon seperated list
 * of metrics. Use the `papi_native_avail` tool to find the name of all
 * available metrics.
 */
struct PAPIMetricCollectorNode final : public MetricCollectorNode {
  explicit PAPIMetricCollectorNode(Array<DeviceWrapper> devices) {
    if (!PAPI_is_initialized()) {
      PAPI_CALL(PAPI_library_init(PAPI_VER_CURRENT));
    }

    // create event sets for each device
    for (auto wrapped_device : devices) {
      Device device = wrapped_device->device;
      int cidx = component_for_device(device);
      // unknown device, skipping
      if (cidx < 0) {
        continue;
      }

      const PAPI_component_info_t* component;
      component = PAPI_get_component_info(cidx);
      if (component->disabled) {
        std::string help_message = "";
        switch (device.device_type) {
          case kDLCPU:
          case kDLCPUPinned:
            help_message =
                "Try setting `sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'`";
            break;
          case kDLGPU:
            help_message =
                "Try enabling gpu profiling with `modprobe nvidia "
                "NVreg_RestrictProfilingToAdminUsers=0`. If that does not work, try adding  "
                "`options nvidia \"NVreg_RestrictProfilingToAdminUsers=0\"` to "
                "`/etc/modprobe.d/nvidia-kernel-common.conf`.";
            break;
          default:
            break;
        }
        LOG(WARNING) << "PAPI could not initialize counters for " << DeviceName(device.device_type)
                     << ": " << component->disabled_reason << "\n"
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

      // load default metrics for device or read them from an environment variable
      std::vector<std::string> metrics;
      std::string dev_name = DeviceName(device.device_type);
      std::transform(dev_name.begin(), dev_name.end(), dev_name.begin(),
                     [](unsigned char c) { return std::toupper(c); });
      const char* env_p =
          std::getenv((std::string("TVM_PAPI_") + dev_name + std::string("_METRICS")).c_str());
      if (env_p != nullptr) {
        std::string metric_string = env_p;
        size_t loc = 0;
        while (loc < metric_string.size()) {
          size_t next = metric_string.find(';', loc);
          if (next == metric_string.npos) {
            next = metric_string.size();
          }
          metrics.push_back(metric_string.substr(loc, next - loc));
          loc = next + 1;
        }
      } else {
        auto it = default_metrics.find(device.device_type);
        if (it != default_metrics.end()) {
          metrics = it->second;
        } else {
          LOG(WARNING) << "No default metrics set for " << dev_name
                       << ". You can specify metrics with the environment variable TVM_PAPI_"
                       << dev_name << "_METRICS.";
        }
      }
      // skip if no metrics exist
      if (metrics.size() == 0) {
        continue;
      }
      papi_metrics[device] = metrics;

      if (static_cast<int>(metrics.size()) > PAPI_num_cmp_hwctrs(cidx)) {
        PAPI_CALL(PAPI_set_multiplex(event_set));
      }

      // add all the metrics
      for (auto metric : metrics) {
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

  /*! \brief Called right before a function call.
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
      std::vector<long_long> values(papi_metrics[dev].size());
      PAPI_CALL(PAPI_read(event_set, values.data()));
      return ObjectRef(make_object<PAPIEventSetNode>(values, dev));
    } else {
      return ObjectRef(nullptr);
    }
  }

  /*! \brief Called right after a function call.
   * \param obj `PAPIEventSetNode` created by a call to `Start`.
   * \returns A mapping from metric name to value.
   */
  Map<String, ObjectRef> Stop(ObjectRef obj) final {
    const PAPIEventSetNode* event_set_node = obj.as<PAPIEventSetNode>();
    std::vector<long_long> end_values(papi_metrics[event_set_node->dev].size());
    PAPI_CALL(PAPI_read(event_sets[event_set_node->dev], end_values.data()));
    std::unordered_map<String, ObjectRef> reported_metrics;
    for (size_t i = 0; i < end_values.size(); i++) {
      reported_metrics[papi_metrics[event_set_node->dev][i]] =
          ObjectRef(make_object<CountNode>(end_values[i] - event_set_node->start_values[i]));
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

  /*! \brief Device-specific event sets. Contains the running counters for that device. */
  std::unordered_map<Device, int> event_sets;
  /*! \brief Device-specific metric names. Order of names matches the order in the corresponding
   * `event_set`. */
  std::unordered_map<Device, std::vector<std::string>> papi_metrics;

  static constexpr const char* _type_key = "PAPIMetricCollectorNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(PAPIMetricCollectorNode, MetricCollectorNode);
};

TVM_REGISTER_GLOBAL("runtime.profiling.metrics.papi")
    .set_body_typed([](Array<DeviceWrapper> devices) {
      return MetricCollector(make_object<PAPIMetricCollectorNode>(devices));
    });

TVM_REGISTER_OBJECT_TYPE(PAPIEventSetNode);
TVM_REGISTER_OBJECT_TYPE(PAPIMetricCollectorNode);

}  // namespace profiling
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_PAPI_PAPI_H_
