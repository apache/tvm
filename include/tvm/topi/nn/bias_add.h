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
 * \brief bias_add op constructions
 * \file nn/bias_add.h
 */
#ifndef TVM_TOPI_NN_BIAS_ADD_H_
#define TVM_TOPI_NN_BIAS_ADD_H_

#include <tvm/te/operation.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/tags.h>
#include <tvm/topi/transform.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

/*!
 * \brief Creates an operation that calculates data + bias
 *
 * \param data Tensor with shape [batch, in_dim]
 * \param bias Tensor with shape [batch].
 * \param axis The axis to add the bias to.
 * \return Tensor with shape [batch, in_dim]
 */
inline tvm::te::Tensor bias_add(const tvm::te::Tensor& data, const tvm::te::Tensor& bias,
                                int axis) {
  int data_ndim = data->shape.size();
  if (axis < 0) {
    axis += data_ndim;
  }
  int num_newaxis = data_ndim - axis - 1;
  return add(data, (num_newaxis ? expand_dims(bias, 1, num_newaxis) : bias));
}
}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_BIAS_ADD_H_
