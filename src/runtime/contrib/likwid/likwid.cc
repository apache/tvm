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
#include <likwid-marker.h>
#include <likwid.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/contrib/likwid.h>
#include <tvm/runtime/threading_backend.h>

namespace tvm {
namespace runtime {
namespace profiling {

struct LikwidMarkerMetricCollectorNode final : public MetricCollectorNode {
  void Init(Array<DeviceWrapper> devices) {
    setenv("LIKWID_FORCE", "1", 1);
    LIKWID_MARKER_INIT;
    TVMBackendParallelLaunch(
        [](int task_id, TVMParallelGroupEnv* penv, void* cdata) {
          LIKWID_MARKER_THREADINIT;
          return 0;
        },
        nullptr, 0);
  }

  ObjectRef Start(Device dev, CallId id) final {
    // Need to prefix region name with something. Region with name "0" is
    // apparently already taken.
    String region_name("call" + std::to_string(id->id));
    LIKWID_MARKER_REGISTER(region_name.c_str());
    // This parallel launch waits for all threads to execute. This may introduce overhead.
    TVMBackendParallelLaunch(
        [](int task_id, TVMParallelGroupEnv* penv, void* cdata) {
          // TODO(tkonolige): markers should be registered first, but we have no way of knowing how
          // many we will need
          LIKWID_MARKER_START(static_cast<const char*>(cdata));
          return 0;
        },
        reinterpret_cast<void*>(const_cast<char*>(region_name.c_str())), 0);
    return std::move(region_name);
  }

  Map<String, ObjectRef> Stop(ObjectRef obj) final {
    if (!obj.defined()) {
      return Map<String, ObjectRef>(nullptr);
    }
    String region_name = Downcast<String>(obj);
    TVMBackendParallelLaunch(
        [](int task_id, TVMParallelGroupEnv* penv, void* cdata) {
          LIKWID_MARKER_STOP(static_cast<const char*>(cdata));
          return 0;
        },
        reinterpret_cast<void*>(const_cast<char*>(region_name.c_str())), 0);
    return Map<String, ObjectRef>(nullptr);
  }

  Map<CallId, Map<String, ObjectRef>> Finish() final {
    LIKWID_MARKER_CLOSE;
    return Map<CallId, Map<String, ObjectRef>>();
  }

  bool SupportsNested() const final { return false; }

  ~LikwidMarkerMetricCollectorNode() {}

  static constexpr const char* _type_key = "runtime.profiling.LikwidMarkerMetricCollector";
  TVM_DECLARE_FINAL_OBJECT_INFO(LikwidMarkerMetricCollectorNode, MetricCollectorNode);
};

class LikwidMarkerMetricCollector : public MetricCollector {
 public:
  LikwidMarkerMetricCollector() { data_ = make_object<LikwidMarkerMetricCollectorNode>(); }
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LikwidMarkerMetricCollector, MetricCollector,
                                            LikwidMarkerMetricCollectorNode);
};

MetricCollector CreateLikwidMarkerMetricCollector() { return LikwidMarkerMetricCollector(); }

TVM_REGISTER_OBJECT_TYPE(LikwidMarkerMetricCollectorNode);

TVM_REGISTER_GLOBAL("runtime.profiling.LikwidMarkerMetricCollector").set_body_typed([]() {
  return LikwidMarkerMetricCollector();
});

struct LikwidMetricCollectorNode final : public MetricCollectorNode {
  explicit LikwidMetricCollectorNode(String group) {
    // Tell Likwid to use the performance counters even if they are already in
    // use. Performance counters are often marked in use when not being used
    // because the previous using process crashed.
    setenv("LIKWID_FORCE", "1", 1);
    topology_init();
    numa_init();
    affinity_init();
    timer_init();
    for (auto core : tvm::runtime::threading::CoresUsed()) {
      cores_.push_back(core);
    }
    int err = perfmon_init(cores_.size(), cores_.data());
    CHECK_GE(err, 0) << "Failed to initialize likwid perfmon";

    gid_ = perfmon_addEventSet(group.c_str());
    CHECK_GE(err, 0) << "Failed to add likwid perfmon event set";

    err = perfmon_setupCounters(gid_);
    CHECK_GE(err, 0) << "Failed to set up likwid perfmon counters";
  }

  void Init(Array<DeviceWrapper> devices) {}

  ObjectRef Start(Device dev, CallId id) final {
    // Likwid only supports CPU and nvidia profiling. The nvidia profiling uses
    // a different interface, so we only support CPU for now.
    if (dev.device_type != DLDeviceType::kDLCPU && dev.device_type != DLDeviceType::kDLCUDAHost) {
      return ObjectRef(nullptr);
    }
    int err = perfmon_startCounters();
    CHECK_GE(err, 0) << "Failed to start likwid perfmon counters";
    return String();  // Need to return some non-null thing to let the profiler know we are timing
  }

  Map<String, ObjectRef> Stop(ObjectRef obj) final {
    int err = perfmon_stopCounters();
    CHECK_GE(err, 0) << "Could not stop likwid perfmon counters";
    perfmon_init_maps();
    perfmon_check_counter_map(0);

    Map<String, ObjectRef> metrics;
    for (int i = 0; i < perfmon_getNumberOfMetrics(gid_); i++) {
      // For now we sum all metrics across threads. This may not be appropriate
      // for some metrics.
      double sum = 0;
      for (auto c : cores_) {
        sum += perfmon_getLastMetric(gid_, i, c);
      }
      metrics.Set(String(perfmon_getMetricName(gid_, i)), ObjectRef(make_object<RateNode>(sum)));
    }

    return metrics;
  }

  Map<CallId, Map<String, ObjectRef>> Finish() final {
    return Map<CallId, Map<String, ObjectRef>>();
  }

  bool SupportsNested() const final {
    // Likwid supports nested regions via the marker api, but this information
    // is not available from a programmatic interface.
    return false;
  }

  ~LikwidMetricCollectorNode() {
    perfmon_finalize();
    timer_finalize();
    affinity_finalize();
    numa_finalize();
    topology_finalize();
  }

  int gid_;
  std::vector<int> cores_;

  static constexpr const char* _type_key = "runtime.profiling.LikwidMetricCollector";
  TVM_DECLARE_FINAL_OBJECT_INFO(LikwidMetricCollectorNode, MetricCollectorNode);
};

class LikwidMetricCollector : public MetricCollector {
 public:
  explicit LikwidMetricCollector(String group) {
    data_ = make_object<LikwidMetricCollectorNode>(group);
  }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LikwidMetricCollector, MetricCollector,
                                        LikwidMetricCollectorNode);
};

MetricCollector CreateLikwidMetricCollector(String group) { return LikwidMetricCollector(group); }

TVM_REGISTER_OBJECT_TYPE(LikwidMetricCollectorNode);

TVM_REGISTER_GLOBAL("runtime.profiling.LikwidMetricCollector").set_body_typed([](String group) {
  return LikwidMetricCollector(group);
});

}  // namespace profiling
}  // namespace runtime
}  // namespace tvm
