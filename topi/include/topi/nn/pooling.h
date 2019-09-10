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
 *  Copyright (c) 2017 by Contributors
 * \brief Pooling op constructions
 * \file nn/pooling.h
 */
#ifndef TOPI_NN_POOLING_H_
#define TOPI_NN_POOLING_H_

#include <string>
#include <vector>

#include "topi/detail/pad_utils.h"
#include "topi/nn.h"
#include "topi/reduction.h"
#include "topi/tags.h"
#include "tvm/ir_pass.h"

namespace topi {
namespace nn {
using namespace tvm;

/*! \brief Pooling type */
enum PoolType : int {
  kAvgPool,
  kMaxPool,
};

/*!
* \brief Perform pooling on height and width dimension of data.
*
* \param x The input tensor
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
* \param height_axis index of the height dimension
* \param width_axis index of the width dimension
* \param count_include_pad Whether include padding in the calculation
*
* \return The output tensor in same layout order
*/
inline Tensor pool_impl(const Tensor& x,
                        const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size,
                        const Array<Expr>& padding_size,
                        PoolType pool_type,
                        bool ceil_mode,
                        const size_t height_axis,
                        const size_t width_axis,
                        bool count_include_pad) {
  CHECK(x->shape.size() >= 2) << "Pooling input must >= 2-D (H, W)";
  CHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 4) << "Pooling padding_size must have 4 elements";

  auto kernel_height = cast(Int(32), kernel_size[0]);
  auto kernel_width = cast(Int(32), kernel_size[1]);
  auto stride_height = cast(Int(32), stride_size[0]);
  auto stride_width = cast(Int(32), stride_size[1]);

  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto pad_top = cast(Int(32), padding_size[0]);
  auto pad_left = cast(Int(32), padding_size[1]);
  auto pad_bottom = cast(Int(32), padding_size[2]);
  auto pad_right = cast(Int(32), padding_size[3]);

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_bottom += stride_height - 1;
    pad_right += stride_width - 1;
  }

  Array<Expr> pad_before(std::vector<Expr>(x->shape.size(), 0));
  pad_before.Set(height_axis, pad_top);
  pad_before.Set(width_axis, pad_left);

  Array<Expr> pad_after(std::vector<Expr>(x->shape.size(), 0));
  pad_after.Set(height_axis, pad_bottom);
  pad_after.Set(width_axis, pad_right);

  auto out_height = tvm::ir::Simplify(
    (height - kernel_height + pad_top + pad_bottom) / stride_height + 1);
  auto out_width = tvm::ir::Simplify(
    (width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_axis, out_height);
  out_shape.Set(width_axis, out_width);

  const int64_t *padding_h0 = as_const_int(pad_top);
  const int64_t *padding_w0 = as_const_int(pad_left);
  const int64_t *padding_h1 = as_const_int(pad_bottom);
  const int64_t *padding_w1 = as_const_int(pad_right);
  const bool do_pad = ((padding_h0 && *padding_h0) || (padding_w0 && *padding_w0)) ||
                      ((padding_h1 && *padding_h1) || (padding_w1 && *padding_w1));

  if (pool_type == kMaxPool) {
    auto temp = do_pad ? pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp") : x;
    return tvm::compute(out_shape, [&](const Array<Var>& output) {
      Array<Expr> indices;
      for (const Var& var : output) indices.push_back(var);
      indices.Set(height_axis, output[height_axis] * stride_height + dheight);
      indices.Set(width_axis, output[width_axis] * stride_width + dwidth);
      return tvm::max(temp(indices), { dheight, dwidth });
    }, "tensor", "pool_max");
  } else if (pool_type == kAvgPool) {
    // Pad the inputs
    auto temp = do_pad ? pad(x, pad_before, pad_after, 0, "pad_temp") : x;

    // TVM compute for summing the pooling window.
    auto pool_sum = tvm::compute(out_shape,
    [&](const Array<Var>& output) {
      Array<Expr> indices;
      for (const Var& var : output) indices.push_back(var);
      indices.Set(height_axis, output[height_axis] * stride_height + dheight);
      indices.Set(width_axis, output[width_axis] * stride_width + dwidth);
      return tvm::sum(temp(indices), { dheight, dwidth });
    }, "tensor", "pool_sum");

    // TVM compute for dividing the reduced window sum by kernel size.
    return tvm::compute(out_shape,
    [&](const Array<Var>& output) {
      Array<Expr> indices;
      for (const Var& var : output) indices.push_back(var);
      if (count_include_pad) {
        return pool_sum(indices) / (kernel_height * kernel_width);
      } else {
        Expr h_start = output[height_axis] * stride_height - pad_top;
        Expr w_start = output[width_axis] * stride_width - pad_left;
        Expr h_end = ir::Min::make(h_start + kernel_height, height);
        Expr w_end = ir::Min::make(w_start + kernel_width, width);
        h_start = ir::Max::make(h_start, make_const(Int(32), 0));
        w_start = ir::Max::make(w_start, make_const(Int(32), 0));
        Expr divide_factor = ir::Max::make((h_end - h_start) * (w_end - w_start),
                                           make_const(Int(32), 1));
        return pool_sum(indices) / divide_factor;
      }
    }, "tensor", kElementWise);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

inline Tensor pool_grad_impl(const Tensor& out_grad, const Tensor& x,
                             const Array<Expr>& kernel_size, const Array<Expr>& stride_size,
                             const Array<Expr>& padding_size, PoolType pool_type, bool ceil_mode,
                             const size_t height_axis, const size_t width_axis,
                             bool count_include_pad) {
  CHECK(out_grad->shape.size() >= 2) << "Pooling grad output must >= 2-D (H, W)";
  CHECK(x->shape.size() >= 2) << "Pooling input must >= 2-D (H, W)";
  CHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 4) << "Pooling padding_size must have 4 elements";

  auto kernel_height = cast(Int(32), kernel_size[0]);
  auto kernel_width = cast(Int(32), kernel_size[1]);
  auto stride_height = cast(Int(32), stride_size[0]);
  auto stride_width = cast(Int(32), stride_size[1]);

  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto pad_top = cast(Int(32), padding_size[0]);
  auto pad_left = cast(Int(32), padding_size[1]);
  auto pad_bottom = cast(Int(32), padding_size[2]);
  auto pad_right = cast(Int(32), padding_size[3]);

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_bottom += stride_height - 1;
    pad_right += stride_width - 1;
  }

  Array<Expr> pad_before(std::vector<Expr>(x->shape.size(), 0));
  pad_before.Set(height_axis, pad_top);
  pad_before.Set(width_axis, pad_left);

  Array<Expr> pad_after(std::vector<Expr>(x->shape.size(), 0));
  pad_after.Set(height_axis, pad_bottom);
  pad_after.Set(width_axis, pad_right);

  auto out_height =
      tvm::ir::Simplify((height - kernel_height + pad_top + pad_bottom) / stride_height + 1);
  auto out_width =
      tvm::ir::Simplify((width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_axis, out_height);
  out_shape.Set(width_axis, out_width);

  const int64_t* padding_h0 = as_const_int(pad_top);
  const int64_t* padding_w0 = as_const_int(pad_left);
  const int64_t* padding_h1 = as_const_int(pad_bottom);
  const int64_t* padding_w1 = as_const_int(pad_right);
  const bool do_pad = ((padding_h0 && *padding_h0) || (padding_w0 && *padding_w0)) ||
                      ((padding_h1 && *padding_h1) || (padding_w1 && *padding_w1));

  if (pool_type == kMaxPool) {
    Array<Expr> ravel_shape{x->shape.begin(), x->shape.end()};
    ravel_shape.Set(height_axis, ravel_shape[height_axis] + pad_top + pad_bottom);
    ravel_shape.Set(width_axis, ravel_shape[width_axis] + pad_left + pad_right);

    auto windowh = tvm::reduce_axis(Range(0, (kernel_height + stride_height - 1) / stride_height));
    auto windoww = tvm::reduce_axis(Range(0, (kernel_width + stride_width - 1) / stride_width));

    auto argmax = MakeArgmaxReducer();
    auto pad_x = do_pad ? pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp") : x;

    auto mp_argmax =
        tvm::compute(out_shape,
                     [&](const Array<Var>& inds) {
                       Array<Expr> window_inds{inds.begin(), inds.end()};
                       window_inds.Set(height_axis, inds[height_axis] * stride_height + dheight);
                       window_inds.Set(width_axis, inds[width_axis] * stride_width + dwidth);
                       auto idx = detail::RavelIndex(window_inds, ravel_shape);
                       return argmax({idx, pad_x(window_inds)}, {dheight, dwidth}, nullptr);
                     },
                     "maxpool_grad_argmax", kCommReduceIdx);

    auto mp_inds = mp_argmax[0];

    return tvm::compute(
        x->shape,
        [&](const Array<Var>& inds) {
          Array<Expr> pad_inds {inds.begin(), inds.end()};
          pad_inds.Set(height_axis, pad_inds[height_axis] + pad_top);
          pad_inds.Set(width_axis, pad_inds[width_axis] + pad_left);
          auto idx = detail::RavelIndex(pad_inds, ravel_shape);

          Array<Expr> out_idx {inds.begin(), inds.end()};
          out_idx.Set(height_axis, (inds[height_axis] + pad_top) / stride_height - windowh);
          out_idx.Set(width_axis, (inds[width_axis] + pad_left) / stride_width - windoww);

          Expr out_idx_lower_h = ir::Select::make(
              pad_inds[height_axis] < kernel_height, make_const(Int(32), 0),
              (pad_inds[height_axis] - kernel_height) / stride_height + 1);
          Expr out_idx_lower_w = ir::Select::make(
              pad_inds[width_axis] < kernel_width, make_const(Int(32), 0),
              (pad_inds[width_axis] - kernel_width) / stride_width + 1);

          return tvm::sum(
              tvm::if_then_else(ir::And::make(
                  ir::And::make(out_idx[height_axis] >= out_idx_lower_h,
                                out_idx[width_axis] >= out_idx_lower_w),
                  mp_inds(out_idx) == idx),
                  out_grad(out_idx), make_const(x->dtype, 0)),
              {windowh, windoww});
        },
        "T_pool_grad", "pool_grad_max");
  } else if (pool_type == kAvgPool) {
    auto windowh = tvm::reduce_axis(Range(0, (kernel_height + stride_height - 1) / stride_height));
    auto windoww = tvm::reduce_axis(Range(0, (kernel_width + stride_width - 1) / stride_width));
    return tvm::compute(
        x->shape,
        [&](const Array<Var>& inds) {
          Expr pad_h_idx = inds[height_axis] + pad_top;
          Expr pad_w_idx = inds[width_axis] + pad_left;

          // output indices whose pooling windows cover current input element (can be out-of-bound)
          Array<Expr> out_idx{inds.begin(), inds.end()};
          out_idx.Set(height_axis, (pad_h_idx / stride_height - windowh));
          out_idx.Set(width_axis, (pad_w_idx / stride_width - windoww));

          Expr out_idx_lower_h = ir::Select::make(pad_h_idx < kernel_height, make_const(Int(32), 0),
                                                  (pad_h_idx - kernel_height) / stride_height + 1);
          Expr out_idx_lower_w = ir::Select::make(pad_w_idx < kernel_width, make_const(Int(32), 0),
                                                  (pad_w_idx - kernel_width) / stride_width + 1);

          Expr divide_factor;  // number of pooled elements
          if (count_include_pad) {
            divide_factor = kernel_height * kernel_width;
          } else {
            Expr h_start = out_idx[height_axis] * stride_height - pad_top;
            Expr w_start = out_idx[width_axis] * stride_width - pad_left;
            Expr h_end = ir::Min::make(h_start + kernel_height, height);
            Expr w_end = ir::Min::make(w_start + kernel_width, width);
            h_start = ir::Max::make(h_start, make_const(Int(32), 0));
            w_start = ir::Max::make(w_start, make_const(Int(32), 0));
            divide_factor =
                ir::Max::make((h_end - h_start) * (w_end - w_start), make_const(Int(32), 1));
          }
          return tvm::sum(tvm::if_then_else(
              ir::And::make(
                ir::And::make(out_idx[height_axis] >= out_idx_lower_h,
                              out_idx[height_axis] < out_height),
                ir::And::make(out_idx[width_axis] >= out_idx_lower_w,
                              out_idx[width_axis] < out_width)),
              out_grad(out_idx) / divide_factor, make_const(out_grad->dtype, 0)),
              {windowh, windoww});
        },
        "T_pool_grad", "pool_grad_avg");
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return Tensor();
  }
}

inline bool find_height_width(const std::string& layout,
                              int* height_axis,
                              int* width_axis) {
  *height_axis = -1, *width_axis = -1;
  int curr_idx = 0;
  for (size_t i = 0; i < layout.size(); ++i) {
    if ((layout[i] >= 'A' && layout[i] <= 'Z') ||
        (layout[i] >= 'a' && layout[i] <= 'z')) {
      if (layout[i] == 'H') {
        if (*height_axis != -1) return false;
        *height_axis = curr_idx;
      } else if (layout[i] == 'W') {
        if (*width_axis != -1) return false;
        *width_axis = curr_idx;
      } else if (layout[i] == 'h' || layout[i] == 'w') {
        // do not support split on height or width, e.g., NCHW16w
        return false;
      }
      ++curr_idx;
    }
  }
  if (*height_axis == -1 || *width_axis == -1) return false;
  return true;
}

/*!
* \brief Perform pooling on height and width dimension of data.
*        It decides the height and width dimension according to the layout string,
*        in which 'W' and 'H' means width and height respectively.
*        Width and height dimension cannot be split.
*        For example, NCHW, NCHW16c, etc. are valid for pool,
*        while NCHW16w, NCHW16h are not.
*        See \a layout for more information of the layout string convention.
* \param x The input tensor.
* \param kernel_size Vector of two ints: {kernel_height, kernel_width}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param pool_type The type of pooling operator
* \param ceil_mode Whether to use ceil when calculating the output size
* \param layout The input layout. Pooling supports any layout as long as 'H' and 'W' appear.
*        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
*        where upper case indicates a dimension and
*        the corresponding lower case (with factor size) indicates the split dimension.
*        For example, NCHW16c can describe a 5-D tensor of
*        [batch_size, channel, height, width, channel_block].
*        (in which factor size `16` will not be used in pooling but for other operators,
*        it can be used to decide the output shape).
*        Since pooling does not care about the factor size of dimensions
*        other than `H` and `W`, one can pass `NCHWc` as well.
* \param  count_include_pad Whether include padding in the calculation when pool_type is 'avg'
*
*
* \return The output tensor in the same layout
*/
inline Tensor pool(const Tensor& x,
                   const Array<Expr>& kernel_size,
                   const Array<Expr>& stride_size,
                   const Array<Expr>& padding_size,
                   PoolType pool_type,
                   bool ceil_mode,
                   const std::string& layout = "NCHW",
                   bool count_include_pad = true) {
  int height_axis = -1, width_axis = -1;
  CHECK(find_height_width(layout, &height_axis, &width_axis))
    << "Unsupported layout " << layout;
  return pool_impl(x, kernel_size, stride_size, padding_size,
                   pool_type, ceil_mode, height_axis, width_axis,
                   count_include_pad);
}

/*!
 * \brief Calculate gradient of pooling on height and width dimension of data.
 *        It decides the height and width dimension according to the layout string,
 *        in which 'W' and 'H' means width and height respectively.
 *        Width and height dimension cannot be split.
 *        For example, NCHW, NCHW16c, etc. are valid for pool,
 *        while NCHW16w, NCHW16h are not.
 *        See \a layout for more information of the layout string convention.
 * \param out_grad The output gradient tensor.
 * \param x The input tensor.
 * \param kernel_size Vector of two ints: {kernel_height, kernel_width}
 * \param stride_size Vector of two ints: {stride_height, stride_width}
 * \param padding_size Vector of two ints: {padding_height, padding_width}
 * \param pool_type The type of pooling operator
 * \param ceil_mode Whether to use ceil when calculating the output size
 * \param layout The input layout. Pooling supports any layout as long as 'H' and 'W' appear.
 *        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
 *        where upper case indicates a dimension and
 *        the corresponding lower case (with factor size) indicates the split dimension.
 *        For example, NCHW16c can describe a 5-D tensor of
 *        [batch_size, channel, height, width, channel_block].
 *        (in which factor size `16` will not be used in pooling but for other operators,
 *        it can be used to decide the output shape).
 *        Since pooling does not care about the factor size of dimensions
 *        other than `H` and `W`, one can pass `NCHWc` as well.
 * \param  count_include_pad Whether include padding in the calculation when pool_type is 'avg'
 *
 *
 * \return The output tensor in the same layout
 */
inline Tensor pool_grad(const Tensor& out_grad, const Tensor& x, const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size, const Array<Expr>& padding_size,
                        PoolType pool_type, bool ceil_mode, const std::string& layout = "NCHW",
                        bool count_include_pad = true) {
  int height_axis = -1, width_axis = -1;
  CHECK(find_height_width(layout, &height_axis, &width_axis)) << "Unsupported layout " << layout;
  return pool_grad_impl(out_grad, x, kernel_size, stride_size, padding_size, pool_type, ceil_mode,
                        height_axis, width_axis, count_include_pad);
}

inline Expr start_index(const Var& out_index,
                        const Expr& odim,
                        const Expr& idim) {
  return out_index * idim / odim;
}

inline Expr end_index(const Var& out_index,
                      const Expr& odim,
                      const Expr& idim) {
  Expr tmp = (out_index + 1) * idim / odim;
  return tvm::ir::Select::make((out_index + 1) * idim % odim == 0,
                               tmp, tmp + 1);
}

/*!
* \brief Perform adaptive pooling on height and width dimension of data.
*
* \param x The input tensor
* \param output_size Vector of two ints: {output_height, output_width}
* \param pool_type The type of pooling operator
* \param height_axis index of the height dimension
* \param width_axis index of the width dimension
*
* \return The output tensor in same layout order
*/
inline Tensor adaptive_pool_impl(const Tensor& x,
                                 const Array<Expr>& output_size,
                                 PoolType pool_type,
                                 const size_t height_axis,
                                 const size_t width_axis) {
  CHECK_EQ(output_size.size(), 2) << "Pooling kernel_size must have 2 elements";

  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto out_height = cast(Int(32), output_size[0]);
  auto out_width = cast(Int(32), output_size[1]);
  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_axis, out_height);
  out_shape.Set(width_axis, out_width);

  if (pool_type == kMaxPool) {
    return tvm::compute(out_shape, [&](const Array<Var>& output) {
      Array<Expr> indices;
      for (const Var& var : output) indices.push_back(var);
      auto i_start_h = start_index(output[height_axis], out_height, height);
      auto i_end_h = end_index(output[height_axis], out_height, height);
      auto i_start_w = start_index(output[width_axis], out_width, width);
      auto i_end_w = end_index(output[width_axis], out_width, width);
      auto dheight = tvm::reduce_axis(Range(0, i_end_h - i_start_h), "rv1");
      auto dwidth = tvm::reduce_axis(Range(0, i_end_w - i_start_w), "rv2");
      indices.Set(height_axis, i_start_h + dheight);
      indices.Set(width_axis, i_start_w + dwidth);
      return tvm::max(x(indices), { dheight, dwidth });  // NOLINT(*)
    }, "tensor", "adaptive_pool_max");
  } else if (pool_type == kAvgPool) {
    return tvm::compute(out_shape, [&](const Array<Var>& output) {
      Array<Expr> indices;
      for (const Var& var : output) indices.push_back(var);
      auto i_start_h = start_index(output[height_axis], out_height, height);
      auto i_end_h = end_index(output[height_axis], out_height, height);
      auto i_start_w = start_index(output[width_axis], out_width, width);
      auto i_end_w = end_index(output[width_axis], out_width, width);
      auto divide_factor = tvm::cast(x->dtype, (i_end_h - i_start_h)
                                               * (i_end_w - i_start_w));
      auto dheight = tvm::reduce_axis(Range(0, i_end_h - i_start_h), "rv1");
      auto dwidth = tvm::reduce_axis(Range(0, i_end_w - i_start_w), "rv2");
      indices.Set(height_axis, i_start_h + dheight);
      indices.Set(width_axis, i_start_w + dwidth);
      return tvm::sum(x(indices) / divide_factor, { dheight, dwidth });
    }, "tensor", "adaptive_pool_avg");
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

/*!
* \brief Adaptively perform pooling on height and width dimension of data.
*        The pooling kernel and stride sizes are automatically chosen for desired output sizes.
*        It decides the height and width dimension according to the layout string,
*        in which 'W' and 'H' means width and height respectively.
*        Width and height dimension cannot be split.
*        For example, NCHW, NCHW16c, etc. are valid for pool,
*        while NCHW16w, NCHW16h are not.
*        See \a layout for more information of the layout string convention.
*
* \param x The input tensor
* \param output_size Vector of two ints: {output_height, output_width}
* \param pool_type The type of pooling operator
* \param layout The input layout. Pooling supports any layout as long as 'H' and 'W' appear.
*        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
*        where upper case indicates a dimension and
*        the corresponding lower case (with factor size) indicates the split dimension.
*        For example, NCHW16c can describe a 5-D tensor of
*        [batch_size, channel, height, width, channel_block].
*        (in which factor size `16` will not be used in pooling but for other operators,
*        it can be used to decide the output shape).
*        Since pooling does not care about the factor size of dimensions
*        other than `H` and `W`, one can pass `NCHWc` as well.
*
* \return The output tensor in same layout order
*/
inline Tensor adaptive_pool(const Tensor& x,
                            const Array<Expr>& output_size,
                            PoolType pool_type,
                            const std::string& layout = "NCHW") {
  int height_axis = -1, width_axis = -1;
  CHECK(find_height_width(layout, &height_axis, &width_axis))
    << "Unsupported layout " << layout;
  return adaptive_pool_impl(x, output_size, pool_type, height_axis, width_axis);
}

/*!
* \brief Perform global pooling on height and width dimension of data.
*        It decides the height and width dimension according to the layout string,
*        in which 'W' and 'H' means width and height respectively.
*        Width and height dimension cannot be split.
*        For example, NCHW, NCHW16c, ... are valid for global_pool,
*        while NCHW16w, NCHW16h are not.
*        See \a layout for more information of the layout string convention.
*
* \param x The input tensor represent as layout
* \param pool_type The type of pooling operator
* \param layout The input layout. global-pooling supports any layout as long as 'H' and 'W' appear.
*        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
*        where upper case indicates a dimension and
*        the corresponding lower case (with factor size) indicates the sub-dimension.
*        For example, `NCHW16c` can describe a 5-D tensor of
*        [batch_size, channel, height, width, channel_block].
*        (in which factor size `16` will not be used in pooling but for other operators,
*        it can be used to decide the output shape).
*        Since pooling does not care about the factor size of
*        dimensions other than `H` and `W`, one can pass `NCHWc` as well.
*
* \return The output tensor in same layout with height and width dimension size of 1.
*         e.g., for NCHW, the output shape will be [batch, channel, 1, 1]
*/
inline Tensor global_pool(const Tensor& x,
                          PoolType pool_type,
                          const std::string& layout = "NCHW") {
  return adaptive_pool(x, Array<Expr>{1, 1}, pool_type, layout);
}

}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_POOLING_H_
