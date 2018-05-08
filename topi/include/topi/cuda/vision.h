/*!
*  Copyright (c) 2018 by Contributors
* \file cuda/vision.h
* \brief CUDA schedule for vision operations
*/
#ifndef TOPI_CUDA_VISION_H_
#define TOPI_CUDA_VISION_H_

#include "tvm/tvm.h"
#include "tvm/build_module.h"
#include "topi/tags.h"
#include "topi/detail/array_utils.h"
#include "topi/contrib/cublas.h"
#include "topi/generic/extern.h"

namespace topi {
using namespace tvm;
namespace cuda {
/*!
* \brief Create a CUDA schedule for region
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_region(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  auto output = outs[0]->op.output(0);
  auto num_thread = 64;

  auto _schedule_softmax = [&](const Operation& softmax_op) {
    auto softmax_inputs = softmax_op->InputTensors();
    auto softmax = softmax_inputs[0];
    auto max_elem = softmax_inputs[1];
    auto expsum = softmax_inputs[2];

    auto block_x = tvm::thread_axis(Range(), "blockIdx.x");
    auto thread_x = tvm::thread_axis(Range(0, num_thread), "threadIdx.x");

    s[max_elem].bind(max_elem->op.as<ComputeOpNode>()->axis[0], block_x);
    auto k = expsum->op.as<ComputeOpNode>()->reduce_axis[0];
    IterVar ko, ki;
    s[expsum].split(k, num_thread, &ko, &ki);
    auto ef = s.rfactor(expsum, ki)[0];

    s[expsum].bind(s[expsum]->op.as<ComputeOpNode>()->axis[0], block_x);
    s[expsum].bind(s[expsum]->op.as<ComputeOpNode>()->reduce_axis[0], thread_x);
    s[ef].compute_at(s[expsum], s[expsum]->op.as<ComputeOpNode>()->reduce_axis[0]);

    s[expsum].set_store_predicate(static_cast<Expr>(thread_x) == 0);
    IterVar tx, xi;
    s[softmax_op].split_by_nparts(softmax_op.as<ComputeOpNode>()->axis[1], num_thread, &tx, &xi);
    s[softmax_op].bind(tx, thread_x);

    return max_elem->op.as<ComputeOpNode>()->InputTensors()[0];
  };

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_injective(op->tag)) {
      if (!detail::contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    } else if (op->tag == "softmax_output") {
      auto tensor = _schedule_softmax(op);
      if (tensor->op->InputTensors().size() > 0) {
        traverse(tensor->op);
      }
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };

  traverse(outs[0]->op);
  auto k = output->op.as<ComputeOpNode>()->axis[0];
  IterVar bx, tx;
  s[output].split(k, num_thread, &bx, &tx);
  s[output].bind(bx, tvm::thread_axis(Range(), "blockIdx.x"));
  s[output].bind(tx, tvm::thread_axis(Range(), "threadIdx.x"));
  return s;
}
}  // namespace cuda
}  // namespace topi
#endif  // TOPI_CUDA_VISION_H_
