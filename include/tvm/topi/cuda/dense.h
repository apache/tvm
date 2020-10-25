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
 * \file cuda/dense.h
 * \brief CUDA schedule for dense operation
 */
#ifndef TVM_TOPI_CUDA_DENSE_H_
#define TVM_TOPI_CUDA_DENSE_H_

#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/contrib/cublas.h>
#include <tvm/topi/detail/array_utils.h>
#include <tvm/topi/generic/extern.h>
#include <tvm/topi/nn/dense.h>
#include <tvm/topi/tags.h>

namespace tvm {
namespace topi {

using namespace tvm::te;

namespace cuda {
/*!
 * \brief Implementation of dense for CUDA backend
 *
 * \param target The target device
 * \param data Tensor with shape [batch, in_dim]
 * \param weight Tensor with shape [out_dim, in_dim]
 * \param bias Tensor with shape [out_dim]. Optional; to omit bias, pass Tensor()
 * \param out_dtype Output data type. Used for mixed precision.
 *
 * \return Tensor with shape [batch, out_dim]
 */
inline tvm::te::Tensor dense_cuda(const Target& target, const tvm::te::Tensor& data,
                                  const tvm::te::Tensor& weight, const tvm::te::Tensor& bias,
                                  const DataType& out_dtype) {
  ICHECK_EQ(data->shape.size(), 2) << "dense requires 2-D data";
  ICHECK_EQ(weight->shape.size(), 2) << "dense requires 2-D weight";
  if (bias.defined()) {
    ICHECK_EQ(bias->shape.size(), 1) << "dense requires 1-D bias";
  }

  auto batch = data->shape[0];
  auto in_dim = data->shape[1];
  auto out_dim = weight->shape[0];

  if (target->GetLibs().count("cublas")) {
    ICHECK_EQ(data->dtype, out_dtype) << "Mixed precision not supported.";
    auto mm = topi::contrib::cublas_matmul(data, weight, false, true);
    if (bias.defined()) {
      mm = tvm::te::compute(
          {batch, out_dim}, [&](Var i, Var j) { return mm(i, j) + bias(j); }, "tensor", kBroadcast);
    }

    return mm;
  } else {
    return topi::nn::dense(data, weight, bias, out_dtype);
  }
}

/*!
 * \brief Create a CUDA schedule for dense
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_dense(const Target& target, const Array<Tensor>& outs) {
  if (target->kind->name == "cuda" && target->GetLibs().count("cublas")) {
    return topi::generic::schedule_extern(target, outs);
  }

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
    if (detail::contains(s->outputs, dense->op)) {
      out = dense;
    } else {
      out = outs[0]->op.output(0);
      s[dense].compute_at(s[out], s[out]->op.as<ComputeOpNode>()->axis[1]);
    }
    s[out].bind(s[out]->op.as<ComputeOpNode>()->axis[0],
                tvm::te::thread_axis(Range(), "blockIdx.y"));
    s[out].bind(s[out]->op.as<ComputeOpNode>()->axis[1],
                tvm::te::thread_axis(Range(), "blockIdx.x"));

    auto tx = s[dense]->op.as<ComputeOpNode>()->reduce_axis[0];
    auto thread_x = tvm::te::thread_axis(Range(), "threadIdx.x");
    s[dense].bind(tx, thread_x);
    s[dense_f].compute_at(s[dense], tx);
    s[dense].set_store_predicate(static_cast<PrimExpr>(thread_x) == 0);
    s[out].set_store_predicate(static_cast<PrimExpr>(thread_x) == 0);
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
}  // namespace tvm
#endif  // TVM_TOPI_CUDA_DENSE_H_
