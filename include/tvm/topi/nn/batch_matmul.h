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
 * \brief Batch matmul op constructions
 * \file nn/batch_matmul.h
 */
#ifndef TVM_TOPI_NN_BATCH_MATMUL_H_
#define TVM_TOPI_NN_BATCH_MATMUL_H_

#include <tvm/te/operation.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Creates an operation that calculates matrix multiplication in batch.
 *
 * \param x Tensor with shape [batch, M, K]
 * \param y Tensor with shape [batch, N, K]
 *
 * \return Tensor with shape [batch, M, N]
 */
inline tvm::te::Tensor batch_matmul(const tvm::te::Tensor& x, const tvm::te::Tensor& y) {
  CHECK_EQ(x->shape.size(), 3) << "batch_matmul requires 3-D data";
  CHECK_EQ(y->shape.size(), 3) << "batch_matmul requires 3-D data";

  auto batch = x->shape[0];
  auto M = x->shape[1];
  auto K = x->shape[2];
  auto N = y->shape[1];

  auto k = tvm::te::reduce_axis(Range(0, K), "k");
  auto result = tvm::te::compute(
      {batch, M, N}, [&](Var b, Var i, Var j) { return tvm::sum(x(b, i, k) * y(b, j, k), {k}); },
      "tensor", "batch_matmul");

  return result;
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_NN_BATCH_MATMUL_H_
