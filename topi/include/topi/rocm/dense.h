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
 * \file rocm/dense.h
 * \brief rocm schedule for dense operation
 */
#ifndef TOPI_ROCM_DENSE_H_
#define TOPI_ROCM_DENSE_H_

#include <topi/tags.h>
#include <tvm/target/generic_func.h>
#include <tvm/te/operation.h>

#include "topi/contrib/rocblas.h"
#include "topi/cuda/dense.h"
#include "topi/detail/array_utils.h"
#include "topi/generic/extern.h"
#include "topi/nn/dense.h"

namespace topi {
using namespace tvm;
using namespace tvm::te;

namespace rocm {
/*!
 * \brief Implementation of dense for rocm backend
 *
 * \param target The target device
 * \param data Tensor with shape [batch, in_dim]
 * \param weight Tensor with shape [out_dim, in_dim]
 * \param bias Tensor with shape [out_dim]. Optional; to omit bias, pass Tensor()
 * \param out_dtype Output data type. Used for mixed precision.
 *
 * \return Tensor with shape [batch, out_dim]
 */
inline tvm::te::Tensor dense_rocm(const Target& target, const tvm::te::Tensor& data,
                                  const tvm::te::Tensor& weight, const tvm::te::Tensor& bias,
                                  const DataType& out_dtype) {
  CHECK_EQ(data->shape.size(), 2) << "dense requires 2-D data";
  CHECK_EQ(weight->shape.size(), 2) << "dense requires 2-D weight";
  if (bias.defined()) {
    CHECK_EQ(bias->shape.size(), 1) << "dense requires 1-D bias";
  }

  auto batch = data->shape[0];
  auto in_dim = data->shape[1];
  auto out_dim = weight->shape[0];

  if (target->GetLibs().count("rocblas")) {
    CHECK_EQ(data->dtype, out_dtype) << "Mixed precision not supported.";
    auto mm = topi::contrib::rocblas_matmul(data, weight, false, true);
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
 * \brief Create a rocm schedule for dense
 *
 * \param target The target to generate a schedule for.
 * \param outs The output tensors.
 *
 * \return A schedule for the given ops.
 */
inline Schedule schedule_dense(const Target& target, const Array<Tensor>& outs) {
  if (target->id->name == "rocm" && target->GetLibs().count("rocblas")) {
    return topi::generic::schedule_extern(target, outs);
  }

  return topi::cuda::schedule_dense(target, outs);
}

}  // namespace rocm
}  // namespace topi
#endif  // TOPI_ROCM_DENSE_H_
