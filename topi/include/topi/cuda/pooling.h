/*!
*  Copyright (c) 2017 by Contributors
* \file cuda/pooling.h
* \brief CUDA schedule for pooling operations
*/
#ifndef TOPI_CUDA_POOLING_H_
#define TOPI_CUDA_POOLING_H_

#include "topi/tags.h"
#include "topi/detail/fuse.h"
#include "topi/detail/array_utils.h"
#include "tvm/tvm.h"
#include "tvm/build_module.h"

namespace topi {
using namespace tvm;

namespace cuda {

/*!
* \brief Create a CUDA schedule for pool
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_pool(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  auto _schedule = [&](const Tensor& padded_input, const Tensor& pool) {
    if (padded_input->op->is_type<ComputeOpNode>()) {
      s[padded_input].compute_inline();
    }
    auto num_thread = target->max_num_threads;
    Tensor out;
    Tensor OL;
    if (detail::contains(s->outputs, pool->op)) {
      out = pool;
      OL = s.cache_write(pool, "local");
    } else {
      out = outs[0]->op.output(0);
      s[pool].set_scope("local");
    }
    auto fused = detail::Fuse(s[out], s[out]->op.as<ComputeOpNode>()->axis);
    IterVar bx, tx;
    s[out].split(fused, num_thread, &bx, &tx);
    s[out].bind(bx, tvm::thread_axis(Range(), "blockIdx.x"));
    s[out].bind(tx, tvm::thread_axis(Range(), "threadIdx.x"));
    if (detail::contains(s->outputs, pool->op)) {
      s[OL].compute_at(s[out], tx);
    } else {
      s[pool].compute_at(s[out], tx);
    }
  };

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_broadcast(op->tag)) {
      if (!detail::contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    } else if (op->tag.rfind("pool", 0) == 0) {
      // If tag starts with pool
      auto padded_input = op->InputTensors()[0];
      auto pool = op.output(0);
      _schedule(padded_input, pool);
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };

  traverse(outs[0]->op);
  return s;
}

/*!
* \brief Create a CUDA schedule for global_pool
*
* \param target The target to generate a schedule for.
* \param outs The output tensors.
*
* \return A schedule for the given ops.
*/
inline Schedule schedule_global_pool(const Target &target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  auto _schedule = [&](const Tensor& pool) {
    auto num_thread = 8;
    auto block_x = tvm::thread_axis(Range(), "blockIdx.x");
    auto block_y = tvm::thread_axis(Range(), "blockIdx.y");
    auto thread_x = tvm::thread_axis(Range(0, num_thread), "threadIdx.x");
    auto thread_y = tvm::thread_axis(Range(0, num_thread), "threadIdx.y");
    Tensor out;
    Tensor OL;
    if (detail::contains(s->outputs, pool->op)) {
      out = pool;
      OL = s.cache_write(pool, "local");
    } else {
      out = outs[0]->op.output(0);
      s[pool].set_scope("local");
    }

    auto i = s[out]->op.as<ComputeOpNode>()->axis[0];
    auto c = s[out]->op.as<ComputeOpNode>()->axis[1];

    IterVar by, ty;
    s[out].split(i, num_thread, &by, &ty);
    IterVar bx, tx;
    s[out].split(c, num_thread, &bx, &tx);
    s[out].reorder({ by, bx, ty, tx });
    s[out].bind(ty, thread_y);
    s[out].bind(tx, thread_x);
    s[out].bind(by, block_y);
    s[out].bind(bx, block_x);

    if (detail::contains(s->outputs, pool->op)) {
      s[OL].compute_at(s[out], tx);
    } else {
      s[pool].compute_at(s[out], tx);
    }
  };

  std::function<void(Operation)> traverse;
  traverse = [&](const Operation& op) {
    // Inline all one-to-one-mapping operators except the last stage (output)
    if (is_broadcast(op->tag)) {
      if (!detail::contains(s->outputs, op)) {
        s[op].compute_inline();
      }
      for (auto tensor : op->InputTensors()) {
        if (tensor->op->InputTensors().size() > 0) {
          traverse(tensor->op);
        }
      }
    } else if (op->tag.rfind("global_pool", 0) == 0) {
      // If tag starts with global_pool
      auto pool = op.output(0);
      _schedule(pool);
    } else {
      LOG(ERROR) << "Unsupported operator " << op->tag;
    }
  };

  traverse(outs[0]->op);
  return s;
}

}  // namespace cuda
}  // namespace topi
#endif  // TOPI_CUDA_POOLING_H_
