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
#ifndef TVM_TOPI_CUDA_SOFTMAX_H_
#define TVM_TOPI_CUDA_SOFTMAX_H_

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
 * \brief Create a CUDA schedule for the given softmax output tensors.
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_softmax(const Target& target, const Array<Tensor>& outs) {
  Array<Operation> out_ops;
  for (auto t : outs) {
    out_ops.push_back(t->op);
  }
  auto s = create_schedule(out_ops);

  auto softmax = outs[0];
  tvm::te::Tensor max_elem;
  tvm::te::Tensor expsum;
  tvm::te::Tensor exp;
  bool has_exp = false;

  auto tag = softmax->op.as<ComputeOpNode>()->tag;
  if (tag == "softmax_output") {
    expsum = softmax->op->InputTensors()[1];
    exp = softmax->op->InputTensors()[0];
    max_elem = s[exp]->op->InputTensors()[1];
    has_exp = true;
  } else if (tag == "log_softmax_output") {
    max_elem = softmax->op->InputTensors()[1];
    expsum = softmax->op->InputTensors()[2];
  } else {
    LOG(ERROR) << "Tag is expected to be softmax_output or log_softmax_output. Got " << tag;
  }

  int num_thread = 64;
  auto block_x = tvm::te::thread_axis(Range(), "blockIdx.x");
  auto thread_x = tvm::te::thread_axis(Range(0, num_thread), "threadIdx.x");

  if (has_exp) {
    s[exp].bind(exp->op.as<ComputeOpNode>()->axis[0], block_x);
  }

  s[max_elem].bind(max_elem->op.as<ComputeOpNode>()->axis[0], block_x);

  auto k = expsum->op.as<ComputeOpNode>()->reduce_axis[0];
  IterVar ko, ki;
  s[expsum].split(k, num_thread, &ko, &ki);
  auto EF = s.rfactor(expsum, ki)[0];
  s[expsum].bind(s[expsum]->op.as<ComputeOpNode>()->axis[0], block_x);
  s[expsum].bind(s[expsum]->op.as<ComputeOpNode>()->reduce_axis[0], thread_x);
  s[EF].compute_at(s[expsum], s[expsum]->op.as<ComputeOpNode>()->reduce_axis[0]);
  s[expsum].set_store_predicate(thread_x->var == 0);

  IterVar tx, xi;
  s[softmax].split_by_nparts(softmax->op.as<ComputeOpNode>()->axis[1], num_thread, &tx, &xi);
  s[softmax].bind(tx, thread_x);

  return s;
}

}  // namespace cuda
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_CUDA_SOFTMAX_H_
