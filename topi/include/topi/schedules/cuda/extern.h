/*!
*  Copyright (c) 2017 by Contributors
* \file cuda/extern.h
* \brief CUDA schedule for extern followed by injective operations
*/
#ifndef TOPI_SCHEDULES_CUDA_EXTERN_H_
#define TOPI_SCHEDULES_CUDA_EXTERN_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace cuda {
Schedule ScheduleOutputForExtern(Target target, Operation op, Schedule sch) {
  auto x = op.output(0);
  auto fused = Fuse(sch[x], sch[x]->op.as<ComputeOpNode>()->axis);
  auto num_thread = target.max_num_threads;
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
Schedule schedule_extern(const Target& target, Array<Tensor> outs) {
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
#endif  // TOPI_SCHEDULES_CUDA_EXTERN_H_
