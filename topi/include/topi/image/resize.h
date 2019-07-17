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
 * \file topi/image/resize.h
 * \brief image resize constructors
 */
#ifndef TOPI_IMAGE_RESIZE_H_
#define TOPI_IMAGE_RESIZE_H_

#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

#include "topi/tags.h"
#include "topi/elemwise.h"
#include "topi/detail/ravel_unravel.h"
#include "topi/detail/constant_utils.h"
#include "tvm/operation.h"
#include "tvm/expr_operator.h"

namespace topi {
namespace image {
using namespace tvm;

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
inline Expr bilinear_sample_nchw(const Tensor& input, const Array<Expr>& indices,
                                 const Expr max_y, const Expr max_x) {
  auto in_y = indices[2];
  auto yf = tvm::floor(in_y);
  auto yc = tvm::cast(Int(32), tvm::ceil(in_y));

  auto y0 = tvm::cast(Int(32), tvm::floor(in_y));
  auto y1 = tvm::if_then_else((yc > max_y), max_y, yc);
  auto y_lerp = in_y - yf;

  auto in_x = indices[3];
  auto xf = tvm::floor(in_x);
  auto xc = tvm::cast(Int(32), tvm::ceil(in_x));

  auto x0 = tvm::cast(Int(32), tvm::floor(in_x));
  auto x1 = tvm::if_then_else((xc > max_x), max_x, xc);
  auto x_lerp = in_x - xf;

  auto A = input(indices[0], indices[1], y0, x0);
  auto B = input(indices[0], indices[1], y0, x1);
  auto C = input(indices[0], indices[1], y1, x0);
  auto D = input(indices[0], indices[1], y1, x1);

  return A * ( 1 - x_lerp) * ( 1 - y_lerp) +
         B * x_lerp * (1 - y_lerp) +
         C * (1 - x_lerp) * y_lerp +
         D * x_lerp * y_lerp;
}

/*!
* \brief Resize given tensor to given shape using nearest neighbour for NHWC
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize_nearest_neighbor_nhwc(const Tensor& input,
                                           const Array<Expr>& shape,
                                           bool align_corners = false,
                                           std::string name = "tensor",
                                           std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(input->shape[0]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);
  out_shape.push_back(input->shape[3]);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    Array<Expr> idx;
    idx.push_back(indices[0]);
    idx.push_back(indices[1] * input->shape[1] / shape[0]);
    idx.push_back(indices[2] * input->shape[2] / shape[1]);
    idx.push_back(indices[3]);

    return input(idx);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using nearest neighbour for NCHW
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize_nearest_neighbor_nchw(const Tensor& input,
                                           const Array<Expr>& shape,
                                           bool align_corners = false,
                                           std::string name = "tensor",
                                           std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(input->shape[0]);
  out_shape.push_back(input->shape[1]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    Array<Expr> idx;
    idx.push_back(indices[0]);
    idx.push_back(indices[1]);
    idx.push_back(indices[2] * input->shape[2] / shape[0]);
    idx.push_back(indices[3] * input->shape[3] / shape[1]);

    return input(idx);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using nearest neighbour for NCHWc
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize_nearest_neighbor_nchwc(const Tensor& input,
                                            const Array<Expr>& shape,
                                            bool align_corners = false,
                                            std::string name = "tensor",
                                            std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(input->shape[0]);
  out_shape.push_back(input->shape[1]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);
  out_shape.push_back(input->shape[4]);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    Array<Expr> idx;
    idx.push_back(indices[0]);
    idx.push_back(indices[1]);
    idx.push_back(indices[2] * input->shape[2] / shape[0]);
    idx.push_back(indices[3] * input->shape[3] / shape[1]);
    idx.push_back(indices[4]);

     return input(idx);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using nearest neighbour
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param layout input layout
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize_nearest_neighbor(const Tensor& input,
                                      const Array<Expr>& shape,
                                      std::string layout = "NCHW",
                                      bool align_corners = false,
                                      std::string name = "tensor",
                                      std::string tag = kInjective) {
  CHECK_EQ(align_corners, false) << "Align corners not supported for nearest neighbour";
  auto base_layout = layout.substr(0, 4);
  if (layout == "NHWC") {
    return resize_nearest_neighbor_nhwc(input, shape, align_corners);
  } else if (layout == "NCHW") {
    return resize_nearest_neighbor_nchw(input, shape, align_corners);
  } else if (base_layout == "NCHW") {
    // NCHWc
    return resize_nearest_neighbor_nchwc(input, shape, align_corners);
  } else {
    LOG(FATAL) << "Unknown layout: " << layout;
    return Tensor();
  }
}

/*!
* \brief Resize given tensor to given shape using bilinear interpolation for NHWC
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize_bilinear_nhwc(const Tensor& input,
                                   const Array<Expr>& shape,
                                   bool align_corners = false,
                                   std::string name = "tensor",
                                   std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(input->shape[0]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);
  out_shape.push_back(input->shape[3]);

  Expr cone = make_const(Int(32), 1);

  auto in_height = as_const_int(input->shape[1]);
  auto in_width = as_const_int(input->shape[2]);
  auto out_height = as_const_int(shape[0]);
  auto out_width = as_const_int(shape[1]);

  Expr y_ratio;
  Expr x_ratio;

  if (!align_corners) {
    y_ratio = make_const(Float(32), (static_cast<float>(*in_height) /
                                     static_cast<float>(*out_height)));
    x_ratio = make_const(Float(32), (static_cast<float>(*in_width) /
                                     static_cast<float>(*out_width)));
  } else {
    y_ratio = make_const(Float(32), (static_cast<float>(*in_height - 1) /
                                     static_cast<float>(*out_height - 1)));
    x_ratio = make_const(Float(32), (static_cast<float>(*in_width - 1) /
                                     static_cast<float>(*out_width - 1)));
  }

  Expr other_y = tvm::ir::Simplify(input->shape[1] - cone);
  Expr other_x = tvm::ir::Simplify(input->shape[2] - cone);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    auto in_y = indices[1] * y_ratio;
    auto yf = tvm::floor(in_y);
    auto yc = tvm::cast(Int(32), tvm::ceil(in_y));

    auto y0 = tvm::cast(Int(32), tvm::floor(in_y));
    auto y1 = tvm::if_then_else((yc > other_y), other_y, yc);
    auto y_lerp  = in_y - yf;

    auto in_x = indices[2] * x_ratio;
    auto xf = tvm::floor(in_x);
    auto xc = tvm::cast(Int(32), tvm::ceil(in_x));

    auto x0 = tvm::cast(Int(32), tvm::floor(in_x));
    auto x1 = tvm::if_then_else((xc > other_x), other_x, xc);
    auto x_lerp  = in_x - xf;

    auto A = input(indices[0], y0, x0, indices[3]);
    auto B = input(indices[0], y0, x1, indices[3]);
    auto C = input(indices[0], y1, x0, indices[3]);
    auto D = input(indices[0], y1, x1, indices[3]);

    return A * ( 1 - x_lerp) * ( 1 - y_lerp) +
           B * x_lerp * (1 - y_lerp) +
           C * (1 - x_lerp) * y_lerp +
           D * x_lerp * y_lerp;
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using bilinear interpolation for NCHW
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize_bilinear_nchw(const Tensor& input,
                                   const Array<Expr>& shape,
                                   bool align_corners = false,
                                   std::string name = "tensor",
                                   std::string tag = kInjective) {
  Array<Expr> out_shape;
  out_shape.push_back(input->shape[0]);
  out_shape.push_back(input->shape[1]);
  out_shape.push_back(shape[0]);
  out_shape.push_back(shape[1]);

  Expr cone = make_const(Int(32), 1);

  auto in_height = as_const_int(input->shape[2]);
  auto in_width = as_const_int(input->shape[3]);
  auto out_height = as_const_int(shape[0]);
  auto out_width = as_const_int(shape[1]);

  Expr y_ratio;
  Expr x_ratio;

  if (!align_corners) {
    y_ratio = make_const(Float(32), (static_cast<float>(*in_height) /
                                     static_cast<float>(*out_height)));
    x_ratio = make_const(Float(32), (static_cast<float>(*in_width) /
                                     static_cast<float>(*out_width)));
  } else {
    y_ratio = make_const(Float(32), (static_cast<float>(*in_height - 1) /
                                     static_cast<float>(*out_height - 1)));
    x_ratio = make_const(Float(32), (static_cast<float>(*in_width - 1) /
                                     static_cast<float>(*out_width - 1)));
  }

  Expr other_y = tvm::ir::Simplify(input->shape[2] - cone);
  Expr other_x = tvm::ir::Simplify(input->shape[3] - cone);

  return compute(
    out_shape, [&](const Array<Var>& indices) {
    auto in_y = indices[2] * y_ratio;
    auto in_x = indices[3] * x_ratio;
    return bilinear_sample_nchw(input, {indices[0], indices[1], in_y, in_x}, other_y, other_x);
    }, name, tag);
}

/*!
* \brief Resize given tensor to given shape using bilinear interpolation
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param layout input layout
* \param align_corners To preserve centers of 4 corner pixels
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize_bilinear(const Tensor& input,
                              const Array<tvm::Expr>& shape,
                              std::string layout = "NCHW",
                              bool align_corners = false,
                              std::string name = "tensor",
                              std::string tag = kInjective) {
  Tensor ret;

  if (layout == "NHWC") {
    ret = resize_bilinear_nhwc(input, shape, align_corners);
  } else {
    ret = resize_bilinear_nchw(input, shape, align_corners);
  }

  return cast(ret, input->dtype);
}

/*!
* \brief Resize given tensor to given shape
*
* \param input The input tensor.
* \param shape Output shape to resize to.
* \param layout input layout
* \param align_corners To preserve centers of 4 corner pixels
* \param mode Algorithm to use (NEAREST_NEIGHBOR / BILINEAR)
* \param name Name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor resized to given shape
*/
inline Tensor resize(const Tensor& input,
                     const Array<Expr>& shape,
                     std::string layout = "NCHW",
                     bool align_corners = false,
                     std::string mode = "BILINEAR",
                     std::string name = "tensor",
                     std::string tag = kInjective) {
  if (mode == "NEAREST_NEIGHBOR") {
    return resize_nearest_neighbor(input, shape, layout, align_corners);
  } else {
    return resize_bilinear(input, shape, layout, align_corners);
  }
}

}  // namespace image
}  // namespace topi
#endif  // TOPI_IMAGE_RESIZE_H_
