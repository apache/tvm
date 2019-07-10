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
 * \file cuda/injective.h
 * \brief CUDA schedule for injective operations
 */
#ifndef TOPI_CUDA_INJECTIVE_H_
#define TOPI_CUDA_INJECTIVE_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/operation.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace cuda {
/*!
* \brief Schedule a given injective operation.
*
* \param target The target to generate a schedule for.
* \param op The operation representing the injective operation.
* \param s The schedule to apply this scheduling to
*/
inline void ScheduleInjectiveOp(const Target &target, Operation op, Schedule s) {
  auto x = op.output(0);
  auto fused = detail::Fuse(s[x], s[x]->op.as<ComputeOpNode>()->axis);
  auto num_thread = target->max_num_threads;
  IterVar bx, tx;
  s[x].split(fused, num_thread, &bx, &tx);
  s[x].bind(bx, thread_axis(Range(), "blockIdx.x"));
  s[x].bind(tx, thread_axis(Range(), "threadIdx.x"));
}

/*!
 * \brief Create a CUDA schedule for the given output tensors.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_injective(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  tvm::schedule::AutoInlineInjective(s);
  for (auto out : outs) {
    ScheduleInjectiveOp(target, out->op, s);
  }
  return s;
}

}  // namespace cuda
}  // namespace topi
#endif  // TOPI_CUDA_INJECTIVE_H_
