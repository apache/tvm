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
 * \brief Performance counters for profiling via the Likwid library.
 */
#ifndef TVM_RUNTIME_CONTRIB_LIKWID_H_
#define TVM_RUNTIME_CONTRIB_LIKWID_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/profiling.h>

namespace tvm {
namespace runtime {
namespace profiling {

/*! \brief Construct a metric collector that collects data from hardware
 * performance counters using Likwid (https://github.com/RRZE-HPC/likwid)
 *
 * \param group Name of the group to collect performance counters from.
 * `likwid-perfctr -a` will list available groups.
 */
TVM_DLL MetricCollector CreateLikwidMetricCollector(String group);

/*! \brief Construct a metric collector that adds Likwid markers to function
 * calls. This reports no information on its own, but can be use in combination
 * with `likwid-perfctr -m` to report performance metrics at runtime.
 */
TVM_DLL MetricCollector CreateLikwidMarkerMetricCollector();
}  // namespace profiling
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_LIKWID_H_
