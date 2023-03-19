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
 * \brief layer normalization op constructions
 * \file nn/layer_norm.h
 */
#ifndef TVM_TOPI_NN_LAYER_NORM_H_
#define TVM_TOPI_NN_LAYER_NORM_H_

#include <tvm/te/operation.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Layer normalization.
 * \param data N-D tensor with shape [d_0, d_1, ..., d_{N-1}]
 * \param gamma K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where K == len(axis) and
 *              d_{axis_k} == r_k
 * \param beta Optional, K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where
 *             d_{axis_k} == r_k
 * \param axis The axis to normalize over.
 * \param epsilon The epsilon value to avoid division by zero.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 * \return The normalized tensor, with the same shape as data.
 */
inline Tensor layer_norm(const Tensor& data, const Tensor& gamma, const Tensor& beta,
                         const Array<Integer>& axis, double epsilon,
                         std::string name = "T_layer_norm", std::string tag = kInjective) {
  const auto& data_type = data->dtype;
  const auto& gamma_type = gamma.defined() ? gamma->dtype : data_type;
  const auto& beta_type = beta.defined() ? beta->dtype : data_type;
  ICHECK(data_type == gamma_type && data_type == beta_type)
      << "layer_norm: data, gamma and beta must have the same type";
  ICHECK(data_type == DataType::Float(32) || data_type == DataType::Float(16))
      << "layer_norm: only support float32 and float16 for now";
  bool is_float16 = data_type == DataType::Float(16);
  // sum x and x^2
  auto ndim = data->shape.size();
  ICHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis);
  auto reduce_axes = MakeReduceAxes(real_axis, data);
  auto target_shape =
      MakeReduceTargetShape(real_axis, data, /*keepdims=*/false, /*atleast1d=*/true);
  auto func = MakeTupleSumReducer();

  auto compute = [ndim, is_float16, &real_axis, &reduce_axes, &func,
                  &data](const Array<Var>& indices) {
    Array<PrimExpr> eval_range;
    int arg_counter = 0;
    int red_counter = 0;

    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        // real_axis contains i
        eval_range.push_back(reduce_axes[red_counter]);
        red_counter++;
      } else {
        eval_range.push_back(indices[arg_counter]);
        arg_counter++;
      }
    }
    auto square = [is_float16](const PrimExpr& x) {
      if (is_float16) {
        return Cast(DataType::Float(32), x) * Cast(DataType::Float(32), x);
      }
      return x * x;
    };
    if (is_float16) {
      return func({Cast(DataType::Float(32), data(eval_range)), square(data(eval_range))},
                  reduce_axes, nullptr);
    } else {
      return func({data(eval_range), square(data(eval_range))}, reduce_axes, nullptr);
    }
  };

  auto temp_x_x2 =
      tvm::te::compute(target_shape, compute, data->op->name + "_red_temp", kCommReduce);

  auto temp_x = temp_x_x2[0];
  auto temp_x2 = temp_x_x2[1];

  auto reduce_extent = make_const(data->dtype, 1);
  for (int i : real_axis) {
    reduce_extent *= data->shape[i];
  }
  auto layer_norm_func = [&](const Array<Var>& indices) {
    Array<Var> reduce_indices, non_reduce_indices;
    for (int i = 0, n = static_cast<int>(indices.size()); i < n; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        reduce_indices.push_back(indices[i]);
      } else {
        non_reduce_indices.push_back(indices[i]);
      }
    }
    auto mean = temp_x(non_reduce_indices) / reduce_extent;
    auto var = temp_x2(non_reduce_indices) / reduce_extent - mean * mean;
    auto layer_norm = (data(indices) - mean) * tvm::rsqrt(var + make_const(var->dtype, epsilon));
    if (is_float16) {
      layer_norm = Cast(DataType::Float(16), layer_norm);
    }
    layer_norm = topi::multiply(layer_norm, gamma(reduce_indices));
    if (beta.defined()) {
      layer_norm = topi::add(layer_norm, beta(reduce_indices));
    }
    return layer_norm;
  };
  return tvm::te::compute(data->shape, layer_norm_func, name, tag);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_NN_LAYER_NORM_H_
