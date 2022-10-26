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
#ifndef TVM_META_SCHEDULE_TRACE_APPLY_H_
#define TVM_META_SCHEDULE_TRACE_APPLY_H_

#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/target/target.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/trace.h>

#include <string>

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Apply the trace from a TIR module whose anchor block is the same but fused elemewise
 * op blocks differ. This function can be used for transferring a trace tuned on a conv2d -> add
 * subgraph to other subgraphs having the same conv2d workload, for example. We call such trace
 * an "anchor trace". Those blocks that are not scheduled by the given anchor trace will be either
 * inlined or parallelized.
 * \param sch The schedule to apply the anchor trace.
 * \param anchor_trace The trace tuned on other subgraph with the same anchor-block workload.
 * \param target The target information needed for inlining and parallelization.
 */
void ScheduleUsingAnchorTrace(tir::Schedule sch, const tir::Trace& anchor_trace,
                              const tvm::Target& target);

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_TRACE_APPLY_H_
