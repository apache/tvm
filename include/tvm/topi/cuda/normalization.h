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
 * \file cuda/normalization.h
 * \brief CUDA schedule for LRN and l2 normalization operations
 */
#ifndef TVM_TOPI_CUDA_NORMALIZATION_H_
#define TVM_TOPI_CUDA_NORMALIZATION_H_

#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

using namespace tvm::te;
namespace cuda {
/*!
 * \brief Create a CUDA schedule for LRN
 * \param outs The output tensors.
 * \return A schedule for the given ops.
 */
inline Schedule schedule_lrn(const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  Schedule s = create_schedule(out_ops);
  int num_thread = 64;
  IterVar block_x = tvm::te::thread_axis(Range(), "blockIdx.x");
  IterVar thread_x = tvm::te::thread_axis(Range(0, num_thread), "threadIdx.x");
  Tensor lrn = outs[0];
  Tensor sqr_sum_up = lrn->op->InputTensors()[1];
  Tensor sqr_sum = sqr_sum_up->op->InputTensors()[0];
  Tensor set_pad = sqr_sum->op->InputTensors()[0];
  s[set_pad].bind(set_pad->op.as<ComputeOpNode>()->axis[0], block_x);
  IterVar rxk = sqr_sum->op.as<ComputeOpNode>()->reduce_axis[0];
  IterVar xko, xki;
  s[sqr_sum].split(rxk, num_thread, &xko, &xki);
  Tensor srf = s.rfactor(sqr_sum, xki)[0];
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

}  // namespace cuda
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_CUDA_NORMALIZATION_H_
