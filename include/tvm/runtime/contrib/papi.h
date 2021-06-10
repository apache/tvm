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

#include <tvm/runtime/profiling.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>

#include <unordered_map>

namespace tvm {
namespace runtime {
namespace profiling {

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
  explicit PAPIMetricCollectorNode(Map<DeviceWrapper, Array<String>> metrics);
  explicit PAPIMetricCollectorNode() {}

  /*! \brief Initialization call.
   * \param device The devices this collector will be running on
   */
  void Init(Array<DeviceWrapper> devices);
  /*! \brief Called right before a function call. Reads starting values of the
   * measured metrics.
   *
   * \param dev The device the function will be run on.
   * \returns A `PAPIEventSetNode` containing values for the counters at the
   * start of the call. Passed to a corresponding `Stop` call.
   */
  ObjectRef Start(Device dev) final;
  /*! \brief Called right after a function call. Reads ending values of the
   * measured metrics. Computes the change in each metric from the
   * corresponding `Start` call.
   *
   * \param obj `PAPIEventSetNode` created by a call to `Start`.
   * \returns A mapping from metric name to value.
   */
  Map<String, ObjectRef> Stop(ObjectRef obj) final;

  ~PAPIMetricCollectorNode() final;

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
  explicit PAPIMetricCollector(Map<DeviceWrapper, Array<String>> metrics);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PAPIMetricCollector, MetricCollector, PAPIMetricCollectorNode);
};
}  // namespace profiling
}  // namespace runtime
}  // namespace tvm

#endif
