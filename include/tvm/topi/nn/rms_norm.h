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
 * \brief root mean square normalization op constructions
 * \file nn/rms_norm.h
 */
#ifndef TVM_TOPI_NN_RMS_NORM_H_
#define TVM_TOPI_NN_RMS_NORM_H_

#include <tvm/te/operation.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Root mean square normalization.
 * \param data N-D tensor with shape [d_0, d_1, ..., d_{N-1}]
 * \param weight K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where K == len(axis) and
 *               d_{axis_k} == r_k
 * \param axis The axis to normalize over.
 * \param epsilon The epsilon value to avoid division by zero.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 * \return The normalized tensor, with the same shape as data.
 */
inline Tensor rms_norm(const Tensor& data, const Tensor& weight, const Array<Integer>& axis,
                       double epsilon, std::string name = "T_rms_norm",
                       std::string tag = kInjective) {
  const auto& data_type = data->dtype;
  const auto& weight_type = weight.defined() ? weight->dtype : data_type;
  ICHECK(data_type == weight_type) << "rms_norm: data and weight must have the same type";

  const auto& data_fp32 = cast(data, DataType::Float(32));
  const auto& weight_fp32 = cast(weight, DataType::Float(32));

  auto square = multiply(data_fp32, data_fp32);
  auto square_sum = sum(square, axis, /*keepdims=*/false, /*atleast1d=*/true);

  auto ndim = data_fp32->shape.size();
  ICHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis);
  auto reduce_extent = make_const(data_fp32->dtype, 1);
  for (int i : real_axis) {
    reduce_extent *= data_fp32->shape[i];
  }
  auto rms_norm_func = [&](const Array<Var>& indices) {
    Array<Var> reduce_indices, non_reduce_indices;
    for (int i = 0, n = static_cast<int>(indices.size()); i < n; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        reduce_indices.push_back(indices[i]);
      } else {
        non_reduce_indices.push_back(indices[i]);
      }
    }
    auto output =
        data_fp32(indices) * weight_fp32(reduce_indices) *
        tvm::rsqrt(square_sum(non_reduce_indices) / reduce_extent + make_const(data_type, epsilon));
    return output;
  };
  auto rms_norm = tvm::te::compute(data_fp32->shape, rms_norm_func, name, tag);
  return cast(rms_norm, data_type);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_NN_RMS_NORM_H_
