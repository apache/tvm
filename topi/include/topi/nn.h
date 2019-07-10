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
 * \brief NN op constructions
 * \file topi/nn.h
 */
#ifndef TOPI_NN_H_
#define TOPI_NN_H_

#include <algorithm>
#include <string>

#include "topi/tags.h"
#include "topi/detail/constant_utils.h"
#include "tvm/ir.h"
#include "tvm/ir_pass.h"
#include "tvm/operation.h"
#include "tvm/expr_operator.h"

namespace topi {
using namespace tvm;
namespace detail {

template <typename T>
tvm::Expr Map(const tvm::Array<tvm::Expr>& exprs, T op) {
  CHECK_GE(exprs.size(), 1);
  tvm::Expr res = exprs[0];
  for (size_t i = 1; i < exprs.size(); ++i) {
    res = op(res, exprs[i]);
  }
  return res;
}

}  // namespace detail

/*!
 * \brief Creates an operation that performs a rectified linear unit
 *
 * \param t The input tensor
 * \param threshold The relu threshold (default 0)
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the relu operation
 */
template <typename T>
inline tvm::Tensor relu(const tvm::Tensor& t,
                        T threshold = static_cast<T>(0),
                        std::string name = "T_relu",
                        std::string tag = kElementWise) {
  return tvm::compute(
      t->shape,
      [&](const tvm::Array<tvm::Var>& i) {
        auto threshold_const = tvm::make_const(t->dtype, threshold);
        return tvm::max(t(i), threshold_const);
      },
      name,
      tag);
}

/*!
* \brief Creates an operation that performs a leaky rectified linear unit
*
* \param t The input tensor
* \param alpha The slope for the small gradient when t < 0
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the leaky relu operation
*/
inline tvm::Tensor leaky_relu(const tvm::Tensor& t,
                              double alpha = 0.1,
                              std::string name = "T_leaky_relu",
                              std::string tag = kElementWise) {
  return tvm::compute(
    t->shape,
    [&](const tvm::Array<tvm::Var>& i) {
      auto value = t(i);
      auto calpha = tvm::make_const(value.type(), alpha);
      return tvm::ir::Select::make(value > 0, value, value * calpha);
    },
    name,
    tag);
}

/*!
 * \brief Creates an operation that performs a parametric rectified linear unit
 *
 * \param x The input data tensor
 * \param slope The channel-wise slope tensor
 * \param axis The axis where the channel data needs to be applied
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the parametric relu operation
 */
inline tvm::Tensor prelu(const tvm::Tensor &x,
                         const tvm::Tensor &slope,
                         const int axis = 1,
                         std::string name = "T_prelu",
                         std::string tag = kBroadcast) {
  CHECK((size_t)axis < x->shape.size()) <<
        "Wrong axis ("  << axis << ")value. ";
  CHECK(topi::detail::GetConstInt(slope->shape[0]) ==
        topi::detail::GetConstInt(x->shape[axis]))
        << "Wrong slope shape received.";

  return tvm::compute(x->shape,
                     [&](const tvm::Array<tvm::Var> &indices) {
                        auto xval = x(indices);
                        return tvm::ir::Select::make(
                            xval > 0,
                            xval,
                            xval * slope(indices[axis]));
                      },
                      name,
                      tag);
}

/*!
 * \brief Creates an operation that performs padding
 *
 * \param t The input tensor
 * \param pad_before An Array of Expr describing the padding before the
 * respective iterator
 * \param pad_after An Array of Expr describing the padding after the
 * respective iterator
 * \param pad_value The value to fill padding elements with
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the padding operation
 *
 * \note
 *  The pad_after Array must either be empty or have the same length as
 *  pad_before
 *  When pad_after is empty, it takes the same values as pad_before (symmetric
 *  padding)
 *  The pad Array applies from the leading dimensions and skips missing
 *  trailing dimensions:
 *
 *      pad(t(i, j, k), {1}, {0}) returns the equivalent operation for
 *          the following pseudocode:
 *              for i in [1, t.shape[0] + 2]:
 *                  for i in [1, t.shape[0] + 2]:
 *                      for i in [1, t.shape[0] + 2]:
 *                         name(i,j,k) =
 *                             (1 <= i <= t.shape[0] + 1) ?
 *                                 t(i-1, j, k) : 0;
 *
 *
 */
inline tvm::Tensor pad(const tvm::Tensor& t,
                       const tvm::Array<tvm::Expr>& pad_before,
                       tvm::Array<tvm::Expr> pad_after = tvm::Array<tvm::Expr>(),
                       Expr pad_value = Expr(),
                       std::string name = "T_pad",
                       std::string tag = kElementWise) {
  if (pad_after.size() < pad_before.size()) {
    for (size_t i = pad_after.size(); i < pad_before.size(); ++i) {
      pad_after.push_back(pad_before[i]);
    }
  }
  CHECK_GE(pad_before.size(), 1);
  CHECK_EQ(pad_before.size(), pad_after.size());
  tvm::Array<tvm::Expr> output_shape;
  for (size_t i = 0; i < t->shape.size(); ++i) {
    if (i >= pad_before.size()) {
      output_shape.push_back(t->shape[i]);
    } else {
      output_shape.push_back(
          tvm::ir::Simplify(t->shape[i] + pad_before[i] + pad_after[i]));
    }
  }

  if (!pad_value.defined()) {
    pad_value = tvm::make_const(t->dtype, 0);
  }

  auto l = [&](tvm::Array<tvm::Var> ovars) {
    tvm::Array<tvm::Expr> indices;
    tvm::Array<tvm::Expr> sel;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      if (i >= pad_before.size()) {
        indices.push_back(ovars[i]);
        continue;
      }
      if (!topi::detail::EqualCheck(pad_before[i], 0)) {
        sel.push_back(ovars[i] >= pad_before[i]);
        indices.push_back(ovars[i] - pad_before[i]);
      } else {
        indices.push_back(ovars[i]);
      }
      if (!topi::detail::EqualCheck(pad_after[i], 0)) {
        sel.push_back(tvm::ir::Simplify(ovars[i] < pad_before[i] + t->shape[i]));
      }
    }
    if (sel.size() != 0) {
      return tvm::if_then_else(
          detail::Map(sel, tvm::ir::And::make), t(indices), pad_value);
    }
    return t(indices);
  };
  return tvm::compute(output_shape, l, name, tag);
}

/*!
 * \brief Creates an operation that performs a 2-D convolution with an
 * NCHW-layout
 *
 * \param I The 4-D input tensor
 * \param W The 4-D weight tensor
 * \param pad_h A static constant padding amount applied to the height of the
 * image, before and after (symmetric padding)
 * \param pad_w A static constant padding amount applied to the width of the
 * image, before and after (symmetric padding)
 * \param stride_h A static constant striding amount applied to the height of
 * the image, before and after (symmetric padding)
 * \param stride_w A static constant strindingamount applied to the width of
 * the image, before and after (symmetric padding)
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the 2-D convolution operation (NCHW
 * layout)
 */
inline tvm::Tensor conv2d_nchw(const tvm::Tensor& I,
                               const tvm::Tensor& W,
                               int pad_h = 0,
                               int pad_w = 0,
                               int stride_h = 1,
                               int stride_w = 1,
                               std::string name = "T_conv2d_nchw",
                               std::string tag = kConv2dNCHW) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::Expr> output_shape{
      I->shape[0],                                            // B
      W->shape[0],                                            // O
      (I->shape[2] - W->shape[2] + 2 * pad_h) / stride_h + 1,  // H
      (I->shape[3] - W->shape[3] + 2 * pad_w) / stride_w + 1   // W
  };
  auto i = tvm::reduce_axis(tvm::Range{0, I->shape[1]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[2]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[3]}, "kw");
  auto T = (pad_h == 0 && pad_w == 0)
               ? I
               : pad(I, {tvm::Expr(0), tvm::Expr(0), pad_h, pad_w});
  auto l = [&](tvm::Var b, tvm::Var o, tvm::Var h, tvm::Var w) {
    return tvm::sum(
        T(b, i, stride_h * h + kh, stride_w * w + kw) * W(i, o, kh, kw),
        {i, kh, kw});
  };
  return tvm::compute(output_shape, l, name, tag);
}

/*!
 * \brief Creates an operation for 2-D convolution layer with an HWCN-layout
 *
 * \param I The 4-D input tensor
 * \param W The 4-D weight tensor
 * \param pad_h A static constant padding amount applied to the height of the
 * image, before and after (symmetric padding)
 * \param pad_w A static constant padding amount applied to the width of the
 * image, before and after (symmetric padding)
 * \param stride_h A static constant striding amount applied to the height of
 * the image, before and after (symmetric padding)
 * \param stride_w A static constant strindingamount applied to the width of
 * the image, before and after (symmetric padding)
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the 2-D convolution operation
 * (HWCN layout)
 */
inline tvm::Tensor conv2d_hwcn(const tvm::Tensor& I,
                               const tvm::Tensor& W,
                               int pad_h = 0,
                               int pad_w = 0,
                               int stride_h = 1,
                               int stride_w = 1,
                               std::string name = "T_conv2d_hwcn",
                               std::string tag = kConv2dHWCN) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::Expr> output_shape{
      (I->shape[2] - W->shape[2] + 2 * pad_h) / stride_h + 1,  // H
      (I->shape[3] - W->shape[3] + 2 * pad_w) / stride_w + 1,  // W
      I->shape[2],                                             // B
      W->shape[3]                                              // O
  };
  auto i = tvm::reduce_axis(tvm::Range{0, I->shape[3]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[0]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[1]}, "kw");
  auto T = (pad_h == 0 && pad_w == 0) ? I : pad(I, {pad_h, pad_w});
  auto l = [&](tvm::Var b, tvm::Var o, tvm::Var h, tvm::Var w) {
    return tvm::sum(
        T(stride_h * h + kh, stride_w * w + kw, i, b) * W(kh, kw, i, o),
        {i, kh, kw});
  };
  return tvm::compute(output_shape, l, name, tag);
}


/*!
 * \brief Creates an operation that performs a 2-D depthwise convolution with
 * an NCHW-layout
 *
 * \param I The 4-D input tensor
 * \param W The 4-D weight tensor
 * \param pad_h A static constant padding amount applied to the height of the
 * image, before and after (symmetric padding)
 * \param pad_w A static constant padding amount applied to the width of the
 * image, before and after (symmetric padding)
 * \param stride_h A static constant striding amount applied to the height of
 * the image, before and after (symmetric padding)
 * \param stride_w A static constant strindingamount applied to the width of
 * the image, before and after (symmetric padding)
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the 2-D depthwise convolution operation
 * (NCHW layout)
 */
inline tvm::Tensor depthwise_conv2d_nchw(const tvm::Tensor& I,
                                         const tvm::Tensor& W,
                                         int pad_h = 0,
                                         int pad_w = 0,
                                         int stride_h = 1,
                                         int stride_w = 1,
                                         std::string name = "T_depthwise_conv2d_nchw",
                                         std::string tag = kDepthwiseConv2dNCHW) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  auto pCM = W->shape[1];  // channel_multiplier
  tvm::Array<tvm::Expr> output_shape{
      I->shape[0],                                            // B
      W->shape[1],                                            // O
      (I->shape[2] - W->shape[2] + 2 * pad_h) / stride_h + 1,  // H
      (I->shape[3] - W->shape[3] + 2 * pad_w) / stride_w + 1   // W
  };
  auto i = tvm::reduce_axis(tvm::Range{0, I->shape[1]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[2]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[3]}, "kw");
  auto T = (pad_h == 0 && pad_w == 0)
               ? I
               : pad(I, {tvm::Expr(0), tvm::Expr(0), pad_h, pad_w});
  auto l = [&](tvm::Var b, tvm::Var o, tvm::Var h, tvm::Var w) {
    return tvm::sum(T(b, i / pCM, stride_h * h + kh, stride_w * w + kw) *
                        W(i / pCM, o % pCM, kh, kw),
                    {i, kh, kw});
  };
  return tvm::compute(output_shape, l, name, tag);
}

inline tvm::Tensor depthwise_conv2d_nhwc(const tvm::Tensor& I,
                                         const tvm::Tensor& W,
                                         int pad_h = 0,
                                         int pad_w = 0,
                                         int stride_h = 1,
                                         int stride_w = 1,
                                         std::string name = "T_depthwise_conv2d_nhwc",
                                         std::string tag = kDepthwiseConv2dNHWC) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[1];
  auto pW = I->shape[2];
  auto pCM = W->shape[1];  // channel_multiplier
  tvm::Array<tvm::Expr> output_shape{
      I->shape[0],                                            // B
      (I->shape[1] - W->shape[1] + 2 * pad_h) / stride_h + 1,  // H
      (I->shape[2] - W->shape[2] + 2 * pad_w) / stride_w + 1,   // W
      W->shape[3],                                            // O
  };
  auto i = tvm::reduce_axis(tvm::Range{0, I->shape[3]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[0]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[1]}, "kw");
  auto T = (pad_h == 0 && pad_w == 0)
               ? I
               : pad(I, {tvm::Expr(0), pad_h, pad_w, tvm::Expr(0)});
  auto l = [&](tvm::Var b, tvm::Var h, tvm::Var w, tvm::Var o) {
    return tvm::sum(T(b, stride_h * h + kh, stride_w * w + kw, i / pCM) *
                        W(kh, kw, i / pCM, o % pCM),
                    {kh, kw, i});
  };
  return tvm::compute(output_shape, l, name, tag);
}

/*!
 * \brief Creates an operation that performs a 2-D group convolution with
 * an NGCHW-layout
 *
 * \param I The 5-D input tensor
 * \param W The 5-D weight tensor
 * \param pad_h A static constant padding amount applied to the height of the
 * image, before and after (symmetric padding)
 * \param pad_w A static constant padding amount applied to the width of the
 * image, before and after (symmetric padding)
 * \param stride_h A static constant striding amount applied to the height of
 * the image, before and after (symmetric padding)
 * \param stride_w A static constant strindingamount applied to the width of
 * the image, before and after (symmetric padding)
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the 2-D groupconvolution operation
 * (NCHW layout)
 */
inline tvm::Tensor group_conv2d_ngchw(const tvm::Tensor& I,
                                      const tvm::Tensor& W,
                                      int pad_h = 0,
                                      int pad_w = 0,
                                      int stride_h = 1,
                                      int stride_w = 1,
                                      std::string name = "T_group_conv2d_ngchw",
                                      std::string tag = kGroupConv2d) {
  CHECK_EQ(5, I->shape.size());
  CHECK_EQ(5, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::Expr> output_shape{
      I->shape[0],                                            // B
      I->shape[1],                                            // G
      W->shape[2],                                            // O
      (I->shape[3] - W->shape[3] + 2 * pad_h) / stride_h + 1,  // H
      (I->shape[4] - W->shape[4] + 2 * pad_w) / stride_w + 1   // W
  };
  auto i = tvm::reduce_axis(tvm::Range{0, I->shape[2]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[3]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[4]}, "kw");

  auto T = (pad_h == 0 && pad_w == 0)
               ? I
               : pad(I, {tvm::Expr(0), tvm::Expr(0), tvm::Expr(0), pad_h, pad_w});
  auto l = [&](tvm::Array<tvm::Var> args) {
    tvm::Var b = args[0];
    tvm::Var g = args[1];
    tvm::Var o = args[2];
    tvm::Var h = args[3];
    tvm::Var w = args[4];
    return tvm::sum(
        I(b, g, i, stride_h * h + kh, stride_w * w + kw) * W(g, i, o, kh, kw),
        {i, kh, kw});
  };
  return tvm::compute(output_shape, l, name, tag);
}

}  // namespace topi
#endif  // TOPI_NN_H_
