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
 * \brief local response normalization op constructions
 * \file nn/local_response_norm.h
 */
#ifndef TVM_TOPI_NN_LOCAL_RESPONSE_NORM_H_
#define TVM_TOPI_NN_LOCAL_RESPONSE_NORM_H_

#include <tvm/te/operation.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Local response normalization inference operator
 *
 * \param data The input tensor. 4-D shape NCHW or NHWC
 * \param size Integer to define normalisation window size
 * \param axis Input data layout channel axis
 * \param alpha Float scaling factor
 * \param beta Exponent value
 * \param bias Offset to avoid dividing by zero
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the Local response normalization operation
 */
inline Tensor lrn(const Tensor& data, int size, int axis = 1, float alpha = 0.0001,
                  float beta = 0.75, float bias = 2, std::string name = "tensor",
                  std::string tag = kBroadcast) {
  CHECK_EQ(data->shape.size(), 4) << "LRN requires 4-D input";
  CHECK_EQ(size % 2, 1) << "size should be odd number";
  CHECK(axis == 1 || axis == 3) << "axis should be 1 or 3 for NCHW and NHWC";
  auto input_shape = data->shape;
  Array<PrimExpr> pad_before{0, 0, 0, 0};
  Array<PrimExpr> pad_after{0, 0, 0, 0};
  pad_before.Set(axis, static_cast<PrimExpr>(size / 2));
  pad_after.Set(axis, static_cast<PrimExpr>(size / 2));
  auto pad_data = pad(data, pad_before, pad_after, 0, "pad_data");
  auto rxs = tvm::te::reduce_axis(Range(0, size), "rxs");
  Tensor sqr_sum;
  if (axis == 1) {
    sqr_sum = tvm::te::compute(input_shape, [&](Var i, Var l, Var j, Var k) {
      return tvm::sum(pad_data(i, l + rxs, j, k) * pad_data(i, l + rxs, j, k), {rxs});
    });
  } else if (axis == 3) {
    sqr_sum = tvm::te::compute(input_shape, [&](Var i, Var l, Var j, Var k) {
      return tvm::sum(pad_data(i, l, j, k + rxs) * pad_data(i, l, j, k + rxs), {rxs});
    });
  }
  auto sqrt_sum_up = tvm::te::compute(input_shape, [&](Var i, Var j, Var k, Var l) {
    return tvm::pow(bias + (div(alpha * sqr_sum(i, j, k, l), size)), beta);
  });
  return topi::divide(data, sqrt_sum_up);
}
}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_LOCAL_RESPONSE_NORM_H_
