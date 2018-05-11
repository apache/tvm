/*!
*  Copyright (c) 2018 by Contributors
* \file cuda/nn.h
* \brief CUDA schedule for lrn and l2 normalization operations
*/
#ifndef TOPI_CUDA_NN_H_
#define TOPI_CUDA_NN_H_

#include "tvm/tvm.h"
#include "tvm/build_module.h"
#include "topi/tags.h"

namespace topi {
using namespace tvm;
namespace cuda {
/*!
* \brief Create a CUDA schedule for LRN
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_lrn(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  auto num_thread = 64;
  auto block_x = tvm::thread_axis(Range(), "blockIdx.x");
  auto thread_x = tvm::thread_axis(Range(0, num_thread), "threadIdx.x");
  auto lrn = outs[0];
  auto sqr_sum_up = lrn->op->InputTensors()[1];
  auto sqr_sum = sqr_sum_up->op->InputTensors()[0];
  auto set_pad = sqr_sum->op->InputTensors()[0];
  s[set_pad].bind(set_pad->op.as<ComputeOpNode>()->axis[0], block_x);
  auto rxk = sqr_sum->op.as<ComputeOpNode>()->reduce_axis[0];
  IterVar xko, xki;
  s[sqr_sum].split(rxk, num_thread, &xko, &xki);
  auto srf = s.rfactor(sqr_sum, xki)[0];
  s[sqr_sum].bind(s[sqr_sum]->op.as<ComputeOpNode>()->axis[0], block_x);
  s[sqr_sum].bind(s[sqr_sum]->op.as<ComputeOpNode>()->reduce_axis[0], thread_x);
  s[srf].compute_at(s[sqr_sum], s[sqr_sum]->op.as<ComputeOpNode>()->reduce_axis[0]);
  s[sqr_sum_up].bind(sqr_sum_up->op.as<ComputeOpNode>()->axis[0], block_x);
  IterVar xto, xti;
  s[lrn].split_by_nparts(lrn->op.as<ComputeOpNode>()->axis[1], num_thread, &xto, &xti);
  s[lrn].bind(lrn->op.as<ComputeOpNode>()->axis[0], block_x);
  s[lrn].bind(xto, thread_x);

  return s;
}

/*!
* \brief Create a CUDA schedule for L2 normalization
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_l2norm(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_injective(op->tag) || op->tag == "l2norm") {
      if (!detail::contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    } else if (op->tag == "comm_reduce") {
      ScheduleReduce(target, op, s, false);
      for (auto tensor : op->InputTensors()) {
        traverse(tensor->op);
      }
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };

  traverse(outs[0]->op);
  auto num_thread = 64;
  auto l2norm = outs[0];
  auto block_x = tvm::thread_axis(Range(), "blockIdx.x");
  auto thread_x = tvm::thread_axis(Range(0, num_thread), "threadIdx.x");
  IterVar xto, xti;
  s[l2norm].split_by_nparts(l2norm->op.as<ComputeOpNode>()->axis[1], num_thread, &xto, &xti);
  s[l2norm].bind(l2norm->op.as<ComputeOpNode>()->axis[0], block_x);
  s[l2norm].bind(xto, thread_x);
  return s;
}
}  // namespace cuda
}  // namespace topi
#endif  // TOPI_CUDA_NN_H_



