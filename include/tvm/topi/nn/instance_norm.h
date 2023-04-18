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
 * \brief instance normalization op constructions
 * \file nn/instance_norm.h
 */
#ifndef TVM_TOPI_NN_INSTANCE_NORM_H_
#define TVM_TOPI_NN_INSTANCE_NORM_H_

#include <tvm/te/operation.h>
#include <tvm/topi/nn/layer_norm.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Instance normalization.
 * \param data N-D tensor with shape [d_0, d_1, ..., d_{N-1}]
 * \param gamma K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where K == len(axis) and
 *              d_{axis_k} == r_k
 * \param beta Optional, K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where
 *             d_{axis_k} == r_k
 * \param axis The axis to normalize over (the axis along which mean and variance are
 * computed).
 * \param epsilon The epsilon value to avoid division by zero.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 * \return The normalized tensor, with the same shape as data.
 */
inline Tensor instance_norm(const Tensor& data, const Tensor& gamma, const Tensor& beta,
                            const Array<Integer>& axis, double epsilon,
                            std::string name = "T_instance_norm", std::string tag = kInjective) {
  return layer_norm(data, gamma, beta, axis, epsilon, name, tag);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_NN_INSTANCE_NORM_H_
