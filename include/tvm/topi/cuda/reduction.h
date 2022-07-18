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
 * \file cuda/reduction.h
 * \brief CUDA schedule for reduction operations
 */
#ifndef TVM_TOPI_CUDA_REDUCTION_H_
#define TVM_TOPI_CUDA_REDUCTION_H_

#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/detail/fuse.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

using namespace tvm::te;

namespace cuda {
/*!
 * \brief Schedule a given reduce operation.
 *
 * \param target The target to generate a schedule for.
 * \param op The operation representing the injective operation.
 * \param sch The schedule to apply this scheduling to
 * \param is_idx_reduce Pass true to schedule a reduce op that returns
 * an index, such as argmax or argmin.
 *
 * \return The schedule given by sch
 */
Schedule ScheduleReduce(const Target& target, Operation op, Schedule sch,
                        bool is_idx_reduce = false) {
  Tensor data_out;
  Tensor data_in;

  if (!is_idx_reduce) {
    data_in = op->InputTensors()[0];
    data_out = op.output(0);
  } else {
    data_out = op->InputTensors()[0];
  }

  auto out_stage = sch[data_out];
  ICHECK_GT(out_stage->op.as<ComputeOpNode>()->reduce_axis.size(), 0)
      << "reduce_axis must be greater than zero";

  bool all_reduce;
  int num_thread;
  IterVar block_x, thread_x, thread_y;

  if (out_stage->op.as<ComputeOpNode>()->axis.size() > 0) {
    all_reduce = false;
    num_thread = 32;
    if (target->kind->name == "opencl" || target->kind->name == "metal") {
      // Without this, CL_INVALID_WORK_GROUP_SIZE occurs with python tests.
      // Don't know why.
      num_thread = 16;
    }
    block_x = tvm::te::thread_axis(Range(), "blockIdx.x");
    thread_x = tvm::te::thread_axis(Range(0, num_thread), "threadIdx.x");
    thread_y = tvm::te::thread_axis(Range(0, num_thread), "threadIdx.y");
  } else {
    all_reduce = true;
    num_thread = target->GetAttr<Integer>("max_num_threads").value().IntValue();
    thread_x = tvm::te::thread_axis(Range(0, num_thread), "threadIdx.x");
  }

  auto fused_reduce = detail::Fuse(out_stage, out_stage->op.as<ComputeOpNode>()->reduce_axis);

  IterVar ko, ki;
  out_stage.split(fused_reduce, num_thread, &ko, &ki);
  auto data_out_rf = sch.rfactor(data_out, ki)[0];
  auto tx = out_stage->op.as<ComputeOpNode>()->reduce_axis[0];
  out_stage.bind(tx, thread_x);
  sch[data_out_rf].compute_at(out_stage, tx);

  Tensor real_output;
  Tensor temp_idx_input, temp_val_input;
  if (is_idx_reduce) {
    real_output = op.output(0);
    temp_idx_input = data_out->op.output(0);
    temp_val_input = data_out->op.output(1);
  } else {
    real_output = data_out;
  }

  auto stage_real = sch[real_output];
  if (!all_reduce) {
    // Fuse and split the axis
    auto fused_outer = detail::Fuse(stage_real, stage_real->op.as<ComputeOpNode>()->axis);
    IterVar bx, outer_in;
    stage_real.split(fused_outer, num_thread, &bx, &outer_in);

    // Bind the axes to threads and blocks
    stage_real.bind(outer_in, thread_y);
    stage_real.bind(bx, block_x);
    if (is_idx_reduce) {
      sch[temp_idx_input].compute_at(stage_real, outer_in);
      sch[temp_val_input].compute_at(stage_real, outer_in);
    }
  } else {
    if (is_idx_reduce) {
      sch[temp_idx_input].compute_at(stage_real, stage_real->op.as<ComputeOpNode>()->axis[0]);
      sch[temp_val_input].compute_at(stage_real, stage_real->op.as<ComputeOpNode>()->axis[0]);
    }
  }

  stage_real.set_store_predicate(static_cast<PrimExpr>(thread_x) == 0);
  return sch;
}

/*!
 * \brief Recursively traverse operator inputs, setting injective inputs
 * to be computed inline.
 *
 * \param s The schedule we are building
 * \param op The current op in the traversal
 */
void TraverseBeforeReduce(Schedule s, Operation op) {
  if (op->IsInstance<PlaceholderOpNode>()) {
    return;
  } else if (is_injective(op->tag)) {
    s[op].compute_inline();
    for (auto tensor : op->InputTensors()) {
      TraverseBeforeReduce(s, tensor->op);
    }
  } else {
    LOG(ERROR) << "Unsupported operator " << op->tag;
  }
}

/*!
 * \brief Schedule a reduce op, then invoke TraverseBeforeReduce on each
 * of the op's inputs.
 *
 * \param target The target to generate a schedule for.
 * \param s The schedule we are building
 * \param op The reduce op
 */
void TraverseAfterReduce(const Target& target, Schedule s, Operation op) {
  if (is_broadcast(op->tag)) {
    LOG(ERROR) << "Elementwise op after reduce is not yet supported";
  } else if (op->tag == kCommReduce) {
    ScheduleReduce(target, op, s, false);
    for (auto tensor : op->InputTensors()) {
      TraverseBeforeReduce(s, tensor->op);
    }
  } else if (op->tag == kCommReduceIdx) {
    ScheduleReduce(target, op, s, true);
    for (auto tensor : op->InputTensors()[0]->op->InputTensors()) {
      TraverseBeforeReduce(s, tensor->op);
    }
  } else {
    LOG(ERROR) << "Unsupported operator " << op->tag;
  }
}

/*!
 * \brief Create a CUDA schedule for a reduce operation.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
Schedule schedule_reduce(const Target& target, Array<Tensor> outs) {
  ICHECK_EQ(outs.size(), 1) << "outs must have size 1";
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);
  TraverseAfterReduce(target, s, outs[0]->op);
  return s;
}

}  // namespace cuda
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_CUDA_REDUCTION_H_
