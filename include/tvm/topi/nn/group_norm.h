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
 * \brief group normalization op constructions
 * \file nn/group_norm.h
 */
#ifndef TVM_TOPI_NN_GROUP_NORM_H_
#define TVM_TOPI_NN_GROUP_NORM_H_

#include <tvm/te/operation.h>

#include <algorithm>
#include <string>
#include <vector>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

inline Tensor group_norm(const Tensor& data, const Tensor& gamma, const Tensor& beta,
                         int num_groups, int channel_axis, const Array<Integer>& axes,
                         double epsilon, std::string name = "T_group_norm",
                         std::string tag = kInjective) {
  const auto& data_type = data->dtype;
  const auto& gamma_type = gamma.defined() ? gamma->dtype : data_type;
  const auto& beta_type = beta.defined() ? beta->dtype : data_type;
  ICHECK(data_type == gamma_type && data_type == beta_type)
      << "group_norm: data, gamma and beta must have the same type";
  ICHECK(data_type == DataType::Float(32) || data_type == DataType::Float(16))
      << "group_norm: only support float32 and float16 for now";
  bool is_float16 = data_type == DataType::Float(16);
  // reshape data C -> G, C/G
  int ndim = data->shape.size();
  channel_axis = GetRealAxis(static_cast<int>(ndim), {channel_axis})[0];

  auto shape = data->shape;
  auto group_size = floordiv(shape[channel_axis], num_groups);
  auto new_shape = Array<PrimExpr>();
  for (int i = 0; i < ndim; ++i) {
    if (i == channel_axis) {
      new_shape.push_back(num_groups);
      new_shape.push_back(group_size);
    } else {
      new_shape.push_back(shape[i]);
    }
  }
  Tensor data_reshaped;
  if (is_float16) {
    data_reshaped = cast(reshape(data, new_shape), DataType::Float(32));
  } else {
    data_reshaped = reshape(data, new_shape);
  }
  // reshape gamma and beta, C -> G, C/G, cast to float32 if float16
  Tensor gamma_reshaped;
  if (gamma.defined()) {
    gamma_reshaped = reshape(gamma, {num_groups, group_size});
  }
  Tensor beta_reshaped;
  if (beta.defined()) {
    beta_reshaped = reshape(beta, {num_groups, group_size});
  }

  // get the new axes to normalize after reshape
  std::vector<int> new_axes{channel_axis + 1};
  for (auto axis : axes) {
    int new_axis = GetRealAxis(static_cast<int>(ndim), {axis})[0];
    if (new_axis < channel_axis) {
      new_axes.push_back(new_axis);
    } else if (new_axis > channel_axis) {
      new_axes.push_back(new_axis + 1);
    } else {
      ICHECK(false) << "axes can not contain channel axis";
    }
  }
  std::sort(new_axes.begin(), new_axes.end());

  // sum x and x^2, cast to float32 if float16
  ndim = data_reshaped->shape.size();
  auto reduce_axes = MakeReduceAxes(new_axes, data_reshaped);
  auto target_shape =
      MakeReduceTargetShape(new_axes, data_reshaped, /*keepdims=*/false, /*atleast1d=*/true);
  auto func = MakeTupleSumReducer();

  auto compute = [ndim, &new_axes, &reduce_axes, &func, &data_reshaped](const Array<Var>& indices) {
    Array<PrimExpr> eval_range;
    int arg_counter = 0;
    int red_counter = 0;

    for (int i = 0; i < ndim; ++i) {
      if (std::find(new_axes.begin(), new_axes.end(), i) != new_axes.end()) {
        // new_axes contains i
        eval_range.push_back(reduce_axes[red_counter]);
        red_counter++;
      } else {
        eval_range.push_back(indices[arg_counter]);
        arg_counter++;
      }
    }
    auto square = [](const PrimExpr& x) { return x * x; };
    return func({data_reshaped(eval_range), square(data_reshaped(eval_range))}, reduce_axes,
                nullptr);
  };

  auto temp_x_x2 =
      tvm::te::compute(target_shape, compute, data->op->name + "_red_temp", kCommReduce);

  auto temp_x = temp_x_x2[0];
  auto temp_x2 = temp_x_x2[1];
  auto reduce_extent = make_const(DataType::Float(32), 1);
  for (auto axis : new_axes) {
    reduce_extent *= data_reshaped->shape[axis];
  }
  auto group_norm_func = [&](const Array<Var>& indices) {
    Array<Var> reduce_indices, non_reduce_indices, gamma_indices;
    for (int i = 0, n = static_cast<int>(indices.size()); i < n; ++i) {
      if (std::find(new_axes.begin(), new_axes.end(), i) != new_axes.end()) {
        reduce_indices.push_back(indices[i]);
      } else {
        non_reduce_indices.push_back(indices[i]);
      }
    }
    gamma_indices = {indices[channel_axis], indices[channel_axis + 1]};
    auto mean = temp_x(non_reduce_indices) / reduce_extent;
    auto var = temp_x2(non_reduce_indices) / reduce_extent - mean * mean;
    PrimExpr group_norm =
        (data_reshaped(indices) - mean) * tvm::rsqrt(var + make_const(data->dtype, epsilon));
    if (is_float16) {
      group_norm = Cast(DataType::Float(16), group_norm);
    }
    if (gamma.defined()) {
      group_norm = topi::multiply(group_norm, gamma_reshaped(gamma_indices));
    }
    if (beta.defined()) {
      group_norm = topi::add(group_norm, beta_reshaped(gamma_indices));
    }
    return group_norm;
  };
  auto group_norm_out = tvm::te::compute(data_reshaped->shape, group_norm_func, name, tag);
  auto group_norm_out_reshaped = reshape(group_norm_out, shape);
  return group_norm_out_reshaped;
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_NN_GROUP_NORM_H_
