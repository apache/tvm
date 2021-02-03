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
  auto batch_id = indices[0];
  auto channel_id = indices[1];
  auto in_y = indices[2];
  auto y0 = tvm::cast(DataType::Int(32), tvm::floor(in_y));  // low_h
  auto y1 = tvm::cast(DataType::Int(32), tvm::ceil(in_y));   // high_h

  auto in_x = indices[3];
  auto x0 = tvm::cast(DataType::Int(32), tvm::floor(in_x));  // low_w
  auto x1 = tvm::cast(DataType::Int(32), tvm::ceil(in_x));   // high_w
  auto x_lerp = in_x - x0;
  auto y_lerp = in_y - y0;

  auto bottom = (1 - x_lerp) * (input(batch_id, channel_id, y0, x0));
  auto top = PrimExpr(0);
  bottom += tvm::if_then_else(y1 > max_y && x1 > max_x, 0, 0);  // Just to maintain structure
  top += tvm::if_then_else(y1 > max_y && x1 > max_x, 0, 0);
  bottom += tvm::if_then_else(y1 <= max_y && x1 > max_x, 0, 0);
  top += tvm::if_then_else(y1 <= max_y && x1 > max_x,
                           (1 - x_lerp) * (input(batch_id, channel_id, y1, x0)), 0);
  bottom +=
      tvm::if_then_else(y1 > max_y && x1 <= max_x, x_lerp * input(batch_id, channel_id, y0, x1), 0);
  top += tvm::if_then_else(y1 > max_y && x1 <= max_x, 0, 0);
  bottom += tvm::if_then_else(y1 <= max_y && x1 <= max_x,
                              x_lerp * input(batch_id, channel_id, y0, x1), 0);
  top += tvm::if_then_else(y1 <= max_y && x1 <= max_x,
                           (1 - x_lerp) * input(batch_id, channel_id, y1, x0) +
                               x_lerp * input(batch_id, channel_id, y1, x1),
                           0);
  return (1 - y_lerp) * bottom + y_lerp * top;
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
inline PrimExpr bilinear_sample_nhwc(const Tensor& input, const Array<PrimExpr>& indices,
                                     const PrimExpr max_y, const PrimExpr max_x) {
  auto batch_id = indices[0];
  auto channel_id = indices[3];
  auto in_y = indices[1];
  auto y0 = tvm::cast(DataType::Int(32), tvm::floor(in_y));  // low_h
  auto y1 = tvm::cast(DataType::Int(32), tvm::ceil(in_y));   // high_h

  auto in_x = indices[2];
  auto x0 = tvm::cast(DataType::Int(32), tvm::floor(in_x));  // low_w
  auto x1 = tvm::cast(DataType::Int(32), tvm::ceil(in_x));   // high_w
  auto x_lerp = in_x - x0;
  auto y_lerp = in_y - y0;

  auto bottom = (1 - x_lerp) * (input(batch_id, y0, x0, channel_id));
  auto top = PrimExpr(0);
  bottom += tvm::if_then_else(y1 > max_y && x1 > max_x, 0, 0);  // Just to maintain structure
  top += tvm::if_then_else(y1 > max_y && x1 > max_x, 0, 0);
  bottom += tvm::if_then_else(y1 <= max_y && x1 > max_x, 0, 0);
  top += tvm::if_then_else(y1 <= max_y && x1 > max_x,
                           (1 - x_lerp) * (input(batch_id, y1, x0, channel_id)), 0);
  bottom +=
      tvm::if_then_else(y1 > max_y && x1 <= max_x, x_lerp * input(batch_id, y0, x1, channel_id), 0);
  top += tvm::if_then_else(y1 > max_y && x1 <= max_x, 0, 0);
  bottom += tvm::if_then_else(y1 <= max_y && x1 <= max_x,
                              x_lerp * input(batch_id, y0, x1, channel_id), 0);
  top += tvm::if_then_else(y1 <= max_y && x1 <= max_x,
                           (1 - x_lerp) * input(batch_id, y1, x0, channel_id) +
                               x_lerp * input(batch_id, y1, x1, channel_id),
                           0);
  return (1 - y_lerp) * bottom + y_lerp * top;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_TENSOR_UTILS_H_
