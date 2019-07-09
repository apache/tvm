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
 * \file cuda/extern.h
 * \brief CUDA schedule for extern followed by injective operations
 */
#ifndef TOPI_CUDA_EXTERN_H_
#define TOPI_CUDA_EXTERN_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/operation.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace cuda {
/*!
 * \brief Schedule a given operation representing one of the outputs of an
 * external function which is followed by injective operations.
 *
 * \param target The target to generate a schedule for.
 * \param op The operation representing the output followed by injective operations.
 * \param sch The schedule to apply this scheduling to
 *
 * \return The schedule given by sch
 */
inline Schedule ScheduleOutputForExtern(Target target, Operation op, Schedule sch) {
  auto x = op.output(0);
  auto fused = detail::Fuse(sch[x], sch[x]->op.as<ComputeOpNode>()->axis);
  auto num_thread = target->max_num_threads;
  IterVar bx, tx;
  sch[x].split(fused, num_thread, &bx, &tx);
  sch[x].bind(bx, tvm::thread_axis(Range(), "blockIdx.x"));
  sch[x].bind(tx, tvm::thread_axis(Range(), "threadIdx.x"));
  return sch;
}

/*!
* \brief Schedule an extern op followed by injective operations.
* For example, cudnn kernel + bias add + relu
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the op.
*/
inline Schedule schedule_extern(const Target& target, Array<Tensor> outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  tvm::schedule::AutoInlineInjective(s);
  for (auto out : outs) {
    if (out->op->derived_from<ExternOpNode>()) {
      continue;
    }
    ScheduleOutputForExtern(target, out->op, s);
  }

  return s;
}

}  // namespace cuda
}  // namespace topi
#endif  // TOPI_CUDA_EXTERN_H_
