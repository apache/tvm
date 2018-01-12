/*!
*  Copyright (c) 2017 by Contributors
* \file cuda/dense.h
* \brief CUDA schedule for dense operation
*/
#ifndef TOPI_SCHEDULES_CUDA_DENSE_H_
#define TOPI_SCHEDULES_CUDA_DENSE_H_

#include "tvm/tvm.h"
#include "tvm/build_module.h"
#include "topi/tags.h"
#include "topi/detail/array_utils.h"

namespace topi {
using namespace tvm;

namespace cuda {
/*!
* \brief Create a CUDA schedule for dense
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
Schedule schedule_dense(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  auto _schedule = [&](const Tensor& dense) {
    auto num_thread = 64;
    auto k = dense->op.as<ComputeOpNode>()->reduce_axis[0];
    IterVar ko, kf;
    s[dense].split(k, num_thread, &ko, &kf);
    auto dense_f = s.rfactor(dense, kf)[0];

    Tensor out;
    if (contains(s->outputs, dense->op)) {
      out = dense;
    } else {
      out = outs[0]->op.output(0);
      s[dense].compute_at(s[out], s[out]->op.as<ComputeOpNode>()->axis[1]);
    }
    s[out].bind(s[out]->op.as<ComputeOpNode>()->axis[0], tvm::thread_axis(Range(), "blockIdx.y"));
    s[out].bind(s[out]->op.as<ComputeOpNode>()->axis[1], tvm::thread_axis(Range(), "blockIdx.x"));

    auto tx = s[dense]->op.as<ComputeOpNode>()->reduce_axis[0];
    auto thread_x = tvm::thread_axis(Range(), "threadIdx.x");
    s[dense].bind(tx, thread_x);
    s[dense_f].compute_at(s[dense], tx);
    s[dense].set_store_predicate(static_cast<Expr>(thread_x) == 0);
    s[out].set_store_predicate(static_cast<Expr>(thread_x) == 0);
  };

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_broadcast(op->tag)) {
      if (!contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    } else if (op->tag == "dense") {
      // If tag starts with global_pool
      auto dense = op.output(0);
      _schedule(dense);
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };

  traverse(outs[0]->op);
  return s;
}

}  // namespace cuda
}  // namespace topi
#endif  // TOPI_SCHEDULES_CUDA_DENSE_H_

