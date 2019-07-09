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
 *  Copyright (c) 2017 by Contributors
 * \brief Dense op constructions
 * \file nn/dense.h
 */
#ifndef TOPI_NN_DENSE_H_
#define TOPI_NN_DENSE_H_

#include <string>

#include "topi/tags.h"
#include "tvm/operation.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Creates an operation that calculates data * weight^T + bias
*
* \param data Tensor with shape [batch, in_dim]
* \param weight Tensor with shape [out_dim, in_dim]
* \param bias Tensor with shape [out_dim]. Optional; to omit bias, pass Tensor()
* \param out_dtype Output data type. Used for mixed precision.
*
* \return Tensor with shape [batch, out_dim]
*/
inline tvm::Tensor dense(const tvm::Tensor& data,
                         const tvm::Tensor& weight,
                         const tvm::Tensor& bias,
                         const Type& out_dtype) {
  CHECK_EQ(data->shape.size(), 2) << "dense requires 2-D data";
  CHECK_EQ(weight->shape.size(), 2) << "dense requires 2-D weight";
  if (bias.defined()) {
    CHECK_EQ(bias->shape.size(), 1) << "dense requires 1-D bias";
  }

  auto batch = data->shape[0];
  auto in_dim = data->shape[1];
  auto out_dim = weight->shape[0];

  auto k = tvm::reduce_axis(Range(0, in_dim), "k");
  auto matmul = tvm::compute(
    { batch, out_dim },
    [&](Var i, Var j) {
      return tvm::sum(tvm::cast(out_dtype, data(i, k)) *
                      tvm::cast(out_dtype, weight(j, k)), { k });
    }, "tensor", "dense");

  if (bias.defined()) {
    matmul = tvm::compute(
      { batch, out_dim },
      [&](Var i, Var j) {
        return matmul(i, j) + tvm::cast(out_dtype, bias(j));
      }, "tensor", kBroadcast);
  }

  return matmul;
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_DENSE_H_
