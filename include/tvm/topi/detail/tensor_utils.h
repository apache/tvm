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
 * \file tensor_utils.h
 * \brief Utility functions for handling tensor
 */
#ifndef TVM_TOPI_DETAIL_TENSOR_UTILS_H_
#define TVM_TOPI_DETAIL_TENSOR_UTILS_H_

#include <tvm/te/operation.h>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Check whether input shape has dimension of size 0;
 *
 * \param x Input shape
 *
 * \return True if the input shape is empty.
 */
inline bool is_empty_shape(const Array<PrimExpr>& x) {
  bool is_empty = false;
  for (const auto& dim : x) {
    if (auto int_dim = dim.as<IntImmNode>()) {
      if (int_dim->value == 0) {
        is_empty = true;
        break;
      }
    }
  }
  return is_empty;
}

/*!
 * \brief Sample a point in a tensor using bilinear interpolation.
 *
 * \param input The input tensor.
 * \param indices The index of the target point, which can be fractional
 * \param max_y The maximum of y dimension
 * \param max_x The maximum of x dimension
 *
 * \return The interpolated value in the given index.
 */
inline PrimExpr bilinear_sample_nchw(const Tensor& input, const Array<PrimExpr>& indices,
                                     const PrimExpr max_y, const PrimExpr max_x) {
  auto in_y = indices[2];
  auto yf = tvm::floor(in_y);
  auto yc = tvm::cast(DataType::Int(32), tvm::ceil(in_y));

  auto y0 = tvm::cast(DataType::Int(32), tvm::floor(in_y));
  auto y1 = tvm::if_then_else((yc > max_y), max_y, yc);
  auto y_lerp = in_y - yf;

  auto in_x = indices[3];
  auto xf = tvm::floor(in_x);
  auto xc = tvm::cast(DataType::Int(32), tvm::ceil(in_x));

  auto x0 = tvm::cast(DataType::Int(32), tvm::floor(in_x));
  auto x1 = tvm::if_then_else((xc > max_x), max_x, xc);
  auto x_lerp = in_x - xf;

  auto A = input(indices[0], indices[1], y0, x0);
  auto B = input(indices[0], indices[1], y0, x1);
  auto C = input(indices[0], indices[1], y1, x0);
  auto D = input(indices[0], indices[1], y1, x1);

  return A * (1 - x_lerp) * (1 - y_lerp) + B * x_lerp * (1 - y_lerp) + C * (1 - x_lerp) * y_lerp +
         D * x_lerp * y_lerp;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_TENSOR_UTILS_H_
