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
 * \brief l2 normalization op constructions
 * \file nn/l2_normalize.h
 */
#ifndef TOPI_NN_L2_NORMALIZE_H_
#define TOPI_NN_L2_NORMALIZE_H_

#include <string>
#include <algorithm>
#include "topi/tags.h"
#include "tvm/operation.h"
namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief L2 normalization inference operator
*
* \param data The input tensor. 4-D with shape [batch, channel, height, width]
* \param eps Epsilon to prevent div by 0
* \param axis Axes over the normalization applied
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the l2 normalization operation
*/
inline Tensor l2_normalize(const Tensor& data,
                           float eps,
                           const Array<Integer>& axis,
                           std::string name = "tensor",
                           std::string tag = "l2_normalize") {
  for (size_t i = 0; i < axis.size(); ++i) {
    int ax = topi::detail::GetConstInt(axis[i]);
    CHECK_LT(ax, data->shape.size()) <<
             "Axis " << ax << " exceeds input data dim " <<
             data->shape.size();
  }
  auto input_shape = data->shape;
  Tensor dot_value = topi::power(data, static_cast<float>(2.0));
  Tensor sum_value = topi::sum(dot_value, axis, true);
  Tensor expand_sum = topi::broadcast_to(sum_value, input_shape);
  return topi::divide(data,
                      topi::sqrt(tvm::compute(expand_sum->shape,
                                              [&](const Array<Var>& i){
                                                return (max(expand_sum(i), eps));
                                              }, name, tag)));
}
}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_L2_NORMALIZE_H_
