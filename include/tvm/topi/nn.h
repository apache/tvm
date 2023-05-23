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
 * \brief NN op constructions
 * \file topi/nn.h
 */
#ifndef TVM_TOPI_NN_H_
#define TVM_TOPI_NN_H_

#include <tvm/arith/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/tags.h>
#include <tvm/topi/transform.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace topi {

using namespace tvm::te;

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
inline tvm::te::Tensor relu(const tvm::te::Tensor& t, T threshold = static_cast<T>(0),
                            std::string name = "T_relu", std::string tag = kElementWise) {
  return tvm::te::compute(
      t->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto threshold_const = tvm::tir::make_const(t->dtype, threshold);
        return tvm::max(t(i), threshold_const);
      },
      name, tag);
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
inline tvm::te::Tensor leaky_relu(const tvm::te::Tensor& t, double alpha = 0.1,
                                  std::string name = "T_leaky_relu",
                                  std::string tag = kElementWise) {
  return tvm::te::compute(
      t->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto value = t(i);
        auto calpha = tvm::tir::make_const(value.dtype(), alpha);
        return tvm::tir::Select(value > 0, value, value * calpha);
      },
      name, tag);
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
inline tvm::te::Tensor prelu(const tvm::te::Tensor& x, const tvm::te::Tensor& slope,
                             const int axis = 1, std::string name = "T_prelu",
                             std::string tag = kBroadcast) {
  ICHECK((size_t)axis < x->shape.size()) << "Wrong axis (" << axis << ")value. ";
  ICHECK(topi::detail::GetConstInt(slope->shape[0]) == topi::detail::GetConstInt(x->shape[axis]))
      << "Wrong slope shape received.";

  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        auto xval = x(indices);
        return tvm::tir::Select(xval > 0, xval, xval * slope(indices[axis]));
      },
      name, tag);
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
 * \param pad_mode Padding type to use.
 * "constant" pads with constant_value;
 * "edge" pads using the edge values of the input array;
 * "reflect" pads by reflecting values with respect to the edges.
 * \param dyn_output_shape Output shape of the pad op, default nullptr.
 * You only need to pass this in if the shape was evaluated dynamically.
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
inline tvm::te::Tensor pad(const tvm::te::Tensor& t, const tvm::Array<tvm::PrimExpr>& pad_before,
                           tvm::Array<tvm::PrimExpr> pad_after = tvm::Array<tvm::PrimExpr>(),
                           PrimExpr pad_value = PrimExpr(), std::string name = "T_pad",
                           std::string tag = kElementWise, std::string pad_mode = "constant",
                           const Array<PrimExpr>* dyn_output_shape = nullptr) {
  if (pad_after.size() < pad_before.size()) {
    for (size_t i = pad_after.size(); i < pad_before.size(); ++i) {
      pad_after.push_back(pad_before[i]);
    }
  }

  arith::Analyzer analyzer;
  ICHECK_GE(pad_before.size(), 1);
  ICHECK_EQ(pad_before.size(), pad_after.size());
  tvm::Array<tvm::PrimExpr> pad_before_int32;
  tvm::Array<tvm::PrimExpr> pad_after_int32;

  for (const auto& ele : pad_before) {
    pad_before_int32.push_back(tvm::cast(tvm::DataType::Int(32), ele));
  }
  for (const auto& ele : pad_after) {
    pad_after_int32.push_back(tvm::cast(tvm::DataType::Int(32), ele));
  }

  tvm::Array<tvm::PrimExpr> output_shape;
  if (dyn_output_shape == nullptr) {
    for (size_t i = 0; i < t->shape.size(); ++i) {
      if (i >= pad_before.size()) {
        output_shape.push_back(t->shape[i]);
      } else {
        output_shape.push_back(
            analyzer.Simplify(t->shape[i] + pad_before_int32[i] + pad_after_int32[i]));
      }
    }
  } else {
    for (size_t i = 0; i < dyn_output_shape->size(); i++) {
      output_shape.push_back((*dyn_output_shape)[i]);
    }
  }

  if (!pad_value.defined()) {
    pad_value = tvm::tir::make_const(t->dtype, 0);
  }

  auto l = [&](tvm::Array<tvm::tir::Var> ovars) {
    tvm::Array<tvm::PrimExpr> indices;
    tvm::Array<tvm::PrimExpr> sel;
    tvm::Array<tvm::PrimExpr> pad_idx;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      if (i >= pad_before_int32.size()) {
        indices.push_back(ovars[i]);
        continue;
      }
      if (!topi::detail::EqualCheck(pad_before_int32[i], 0)) {
        sel.push_back(ovars[i] >= pad_before_int32[i]);
        indices.push_back(ovars[i] - pad_before_int32[i]);
      } else {
        indices.push_back(ovars[i]);
      }
      if (!topi::detail::EqualCheck(pad_after_int32[i], 0)) {
        sel.push_back(analyzer.Simplify(ovars[i] < pad_before_int32[i] + t->shape[i]));
      }
      if (pad_mode == "edge") {
        pad_idx.push_back(
            tvm::if_then_else(ovars[i] < pad_before[i], 0,
                              tvm::if_then_else(ovars[i] >= pad_before[i] + t->shape[i],
                                                t->shape[i] - 1, ovars[i] - pad_before[i])));
      } else if (pad_mode == "reflect") {
        pad_idx.push_back(
            tvm::if_then_else(ovars[i] < pad_before[i], pad_before[i] - ovars[i],
                              tvm::if_then_else(ovars[i] >= pad_before[i] + t->shape[i],
                                                t->shape[i] * 2 - ovars[i] + pad_before[i] - 2,
                                                ovars[i] - pad_before[i])));
      }
    }
    if (sel.size() != 0) {
      if (pad_mode == "constant") {
        return tvm::if_then_else(
            foldl([](PrimExpr a, PrimExpr b, Span span) { return tvm::logical_and(a, b, span); },
                  const_true(1), sel),
            t(indices), pad_value);
      } else if (pad_mode == "edge" || pad_mode == "reflect") {
        return tvm::if_then_else(
            foldl([](PrimExpr a, PrimExpr b, Span span) { return tvm::logical_and(a, b, span); },
                  const_true(1), sel),
            t(indices), t(pad_idx));
      }
    }
    return t(indices);
  };
  return tvm::te::compute(output_shape, l, name, tag);
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
inline tvm::te::Tensor conv2d_nchw(const tvm::te::Tensor& I, const tvm::te::Tensor& W,
                                   int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1,
                                   std::string name = "T_conv2d_nchw",
                                   std::string tag = kConv2dNCHW) {
  ICHECK_EQ(4, I->shape.size());
  ICHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::PrimExpr> output_shape{
      I->shape[0],                                                    // B
      W->shape[0],                                                    // O
      indexdiv(I->shape[2] - W->shape[2] + 2 * pad_h, stride_h) + 1,  // H
      indexdiv(I->shape[3] - W->shape[3] + 2 * pad_w, stride_w) + 1   // W
  };
  auto i = tvm::te::reduce_axis(tvm::Range{0, I->shape[1]}, "i");
  auto kh = tvm::te::reduce_axis(tvm::Range{0, W->shape[2]}, "kh");
  auto kw = tvm::te::reduce_axis(tvm::Range{0, W->shape[3]}, "kw");
  auto T =
      (pad_h == 0 && pad_w == 0) ? I : pad(I, {tvm::PrimExpr(0), tvm::PrimExpr(0), pad_h, pad_w});
  auto l = [&](tvm::tir::Var b, tvm::tir::Var o, tvm::tir::Var h, tvm::tir::Var w) {
    return tvm::sum(T(b, i, stride_h * h + kh, stride_w * w + kw) * W(o, i, kh, kw), {i, kh, kw});
  };
  return tvm::te::compute(output_shape, l, name, tag);
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
inline tvm::te::Tensor conv2d_hwcn(const tvm::te::Tensor& I, const tvm::te::Tensor& W,
                                   int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1,
                                   std::string name = "T_conv2d_hwcn",
                                   std::string tag = kConv2dHWCN) {
  ICHECK_EQ(4, I->shape.size());
  ICHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::PrimExpr> output_shape{
      indexdiv(I->shape[2] - W->shape[2] + 2 * pad_h, stride_h) + 1,  // H
      indexdiv(I->shape[3] - W->shape[3] + 2 * pad_w, stride_w) + 1,  // W
      I->shape[2],                                                    // B
      W->shape[3]                                                     // O
  };
  auto i = tvm::te::reduce_axis(tvm::Range{0, I->shape[3]}, "i");
  auto kh = tvm::te::reduce_axis(tvm::Range{0, W->shape[0]}, "kh");
  auto kw = tvm::te::reduce_axis(tvm::Range{0, W->shape[1]}, "kw");
  auto T = (pad_h == 0 && pad_w == 0) ? I : pad(I, {pad_h, pad_w});
  auto l = [&](tvm::tir::Var b, tvm::tir::Var o, tvm::tir::Var h, tvm::tir::Var w) {
    return tvm::sum(T(stride_h * h + kh, stride_w * w + kw, i, b) * W(kh, kw, i, o), {i, kh, kw});
  };
  return tvm::te::compute(output_shape, l, name, tag);
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
inline tvm::te::Tensor depthwise_conv2d_nchw(const tvm::te::Tensor& I, const tvm::te::Tensor& W,
                                             int pad_h = 0, int pad_w = 0, int stride_h = 1,
                                             int stride_w = 1,
                                             std::string name = "T_depthwise_conv2d_nchw",
                                             std::string tag = kDepthwiseConv2dNCHW) {
  ICHECK_EQ(4, I->shape.size());
  ICHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  auto pCM = W->shape[1];  // channel_multiplier
  tvm::Array<tvm::PrimExpr> output_shape{
      I->shape[0],                                                    // B
      W->shape[1],                                                    // O
      indexdiv(I->shape[2] - W->shape[2] + 2 * pad_h, stride_h) + 1,  // H
      indexdiv(I->shape[3] - W->shape[3] + 2 * pad_w, stride_w) + 1   // W
  };
  auto i = tvm::te::reduce_axis(tvm::Range{0, I->shape[1]}, "i");
  auto kh = tvm::te::reduce_axis(tvm::Range{0, W->shape[2]}, "kh");
  auto kw = tvm::te::reduce_axis(tvm::Range{0, W->shape[3]}, "kw");
  auto T =
      (pad_h == 0 && pad_w == 0) ? I : pad(I, {tvm::PrimExpr(0), tvm::PrimExpr(0), pad_h, pad_w});
  auto l = [&](tvm::tir::Var b, tvm::tir::Var o, tvm::tir::Var h, tvm::tir::Var w) {
    return tvm::sum(T(b, indexdiv(i, pCM), stride_h * h + kh, stride_w * w + kw) *
                        W(indexdiv(i, pCM), indexmod(o, pCM), kh, kw),
                    {i, kh, kw});
  };
  return tvm::te::compute(output_shape, l, name, tag);
}

inline tvm::te::Tensor depthwise_conv2d_nhwc(const tvm::te::Tensor& I, const tvm::te::Tensor& W,
                                             int pad_h = 0, int pad_w = 0, int stride_h = 1,
                                             int stride_w = 1,
                                             std::string name = "T_depthwise_conv2d_nhwc",
                                             std::string tag = kDepthwiseConv2dNHWC) {
  ICHECK_EQ(4, I->shape.size());
  ICHECK_EQ(4, W->shape.size());
  auto pH = I->shape[1];
  auto pW = I->shape[2];
  auto pCM = W->shape[1];  // channel_multiplier
  tvm::Array<tvm::PrimExpr> output_shape{
      I->shape[0],                                                    // B
      indexdiv(I->shape[1] - W->shape[1] + 2 * pad_h, stride_h) + 1,  // H
      indexdiv(I->shape[2] - W->shape[2] + 2 * pad_w, stride_w) + 1,  // W
      W->shape[3],                                                    // O
  };
  auto i = tvm::te::reduce_axis(tvm::Range{0, I->shape[3]}, "i");
  auto kh = tvm::te::reduce_axis(tvm::Range{0, W->shape[0]}, "kh");
  auto kw = tvm::te::reduce_axis(tvm::Range{0, W->shape[1]}, "kw");
  auto T =
      (pad_h == 0 && pad_w == 0) ? I : pad(I, {tvm::PrimExpr(0), pad_h, pad_w, tvm::PrimExpr(0)});
  auto l = [&](tvm::tir::Var b, tvm::tir::Var h, tvm::tir::Var w, tvm::tir::Var o) {
    return tvm::sum(T(b, stride_h * h + kh, stride_w * w + kw, indexdiv(i, pCM)) *
                        W(kh, kw, indexdiv(i, pCM), indexmod(o, pCM)),
                    {kh, kw, i});
  };
  return tvm::te::compute(output_shape, l, name, tag);
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
inline tvm::te::Tensor group_conv2d_ngchw(const tvm::te::Tensor& I, const tvm::te::Tensor& W,
                                          int pad_h = 0, int pad_w = 0, int stride_h = 1,
                                          int stride_w = 1,
                                          std::string name = "T_group_conv2d_ngchw",
                                          std::string tag = kGroupConv2d) {
  ICHECK_EQ(5, I->shape.size());
  ICHECK_EQ(5, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::PrimExpr> output_shape{
      I->shape[0],                                                    // B
      I->shape[1],                                                    // G
      W->shape[2],                                                    // O
      indexdiv(I->shape[3] - W->shape[3] + 2 * pad_h, stride_h) + 1,  // H
      indexdiv(I->shape[4] - W->shape[4] + 2 * pad_w, stride_w) + 1   // W
  };
  auto i = tvm::te::reduce_axis(tvm::Range{0, I->shape[2]}, "i");
  auto kh = tvm::te::reduce_axis(tvm::Range{0, W->shape[3]}, "kh");
  auto kw = tvm::te::reduce_axis(tvm::Range{0, W->shape[4]}, "kw");

  auto T = (pad_h == 0 && pad_w == 0)
               ? I
               : pad(I, {tvm::PrimExpr(0), tvm::PrimExpr(0), tvm::PrimExpr(0), pad_h, pad_w});
  auto l = [&](tvm::Array<tvm::tir::Var> args) {
    tvm::tir::Var b = args[0];
    tvm::tir::Var g = args[1];
    tvm::tir::Var o = args[2];
    tvm::tir::Var h = args[3];
    tvm::tir::Var w = args[4];
    return tvm::sum(I(b, g, i, stride_h * h + kh, stride_w * w + kw) * W(g, i, o, kh, kw),
                    {i, kh, kw});
  };
  return tvm::te::compute(output_shape, l, name, tag);
}

/*!
 * \brief Divide spatial dimensions of the input into a grid of blocks.
 *
 * \param data The input tensor.
 * \param block_shape The size of the spatial block.
 * \param pad_before The zero-padding size before each spatial dimension.
 * \param pad_after The zero-padding size after each spatial dimension.
 * \param pad_value The value used for padding.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor whose op member is the space_to_batch_nd operation
 */
inline tvm::te::Tensor space_to_batch_nd(const tvm::te::Tensor& data,
                                         const tvm::Array<Integer>& block_shape,
                                         const tvm::Array<tvm::PrimExpr>& pad_before,
                                         const tvm::Array<tvm::PrimExpr>& pad_after,
                                         PrimExpr pad_value = PrimExpr(),
                                         std::string name = "space_to_batch_nd",
                                         std::string tag = kInjective) {
  tvm::te::Tensor padded_t;
  CHECK_EQ(pad_before.size(), pad_after.size());
  CHECK_EQ(block_shape.size(), pad_before.size())
      << "Paddings must be provided for each spatial dimension";
  tvm::Array<tvm::PrimExpr> pad_before_int32;
  tvm::Array<tvm::PrimExpr> pad_after_int32;

  // pad size for batch dimension is 0
  pad_before_int32.push_back(tvm::cast(tvm::DataType::Int(32), 0));
  pad_after_int32.push_back(tvm::cast(tvm::DataType::Int(32), 0));
  // insert pad sizes given for spatial dimensions
  for (const auto& ele : pad_before) {
    pad_before_int32.push_back(tvm::cast(tvm::DataType::Int(32), ele));
  }
  for (const auto& ele : pad_after) {
    pad_after_int32.push_back(tvm::cast(tvm::DataType::Int(32), ele));
  }

  // pad the input with paddings provided
  if (!pad_value.defined()) {
    pad_value = tvm::tir::make_const(data->dtype, 0);
  }
  padded_t = pad(data, pad_before_int32, pad_after_int32, pad_value);

  auto input_shape = data->shape;
  auto padded_shape = padded_t->shape;

  // infer shapes
  tvm::Array<PrimExpr> r_shape;
  tvm::Array<Integer> axis;
  tvm::Array<PrimExpr> o_shape;

  size_t num_block_dims = block_shape.size();
  int batch = static_cast<int>(GetConstInt(input_shape[0]));
  tvm::PrimExpr block_shape_prod(1);
  r_shape.push_back(batch);

  for (size_t i = 1; i <= num_block_dims; i++) {
    int padded_input = static_cast<int>(GetConstInt(padded_shape[i]));
    int block_size = static_cast<int>(GetConstInt(block_shape[i - 1]));
    CHECK_EQ((padded_input % block_size), 0)
        << "(" << i
        << ")th "
           "Input dimension after padding ("
        << padded_input << ")"
        << " must be divisible by its block size (" << block_size << ")";

    r_shape.push_back(div(padded_shape[i], block_shape[i - 1]));
    r_shape.push_back(block_shape[i - 1]);
    block_shape_prod *= block_shape[i - 1];
    axis.push_back(Integer(r_shape.size() - 1));  // index of block_shape[i - 1]
  }

  size_t n = axis.size();
  axis.push_back(0);  // batch is at index 0
  // index of (padded_shape[i] / block_shape[i - 1]) in r_shape
  for (size_t i = 0; i < n; i++) {
    axis.push_back(static_cast<int>(GetConstInt(axis[i] - 1)));
  }
  o_shape.push_back(tvm::PrimExpr(batch) * block_shape_prod);
  for (size_t i = 1; i <= num_block_dims; i++) {
    o_shape.push_back(div(padded_shape[i], block_shape[i - 1]));
  }
  // append remaining shape
  for (size_t i = num_block_dims + 1; i < input_shape.size(); i++) {
    r_shape.push_back(input_shape[i]);
    axis.push_back(Integer(r_shape.size() - 1));  // index of remaining shape in r_shape
    o_shape.push_back(input_shape[i]);
  }

  tvm::te::Tensor output = reshape(padded_t, r_shape);
  output = transpose(output, axis);
  output = reshape(output, o_shape);

  return output;
}

/*!
 * \brief Reshape the batch dimension into spatial dimensions.
 *
 * \param data The input tensor.
 * \param block_shape The size of the spatial block.
 * \param crop_begin_list The begin crop size for each spatial dimension.
 * \param crop_end_list The end crop size for each spatial dimension.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor whose op member is the batch_to_space_nd operation
 */
inline tvm::te::Tensor batch_to_space_nd(const tvm::te::Tensor& data,
                                         const tvm::Array<Integer>& block_shape,
                                         const tvm::Array<tvm::PrimExpr>& crop_begin_list,
                                         const tvm::Array<tvm::PrimExpr>& crop_end_list,
                                         std::string name = "batch_to_space_nd",
                                         std::string tag = kInjective) {
  // Construct shapes for reshape and transpose operation
  Array<PrimExpr> in_shape = data->shape;
  Array<PrimExpr> r_shape;
  Array<Integer> axis;
  size_t num_block_dims = block_shape.size();
  size_t num_input_dims = in_shape.size();
  tvm::PrimExpr block_shape_prod(1);
  int batch = static_cast<int>(GetConstInt(in_shape[0]));

  for (size_t i = 0; i < num_block_dims; i++) {
    r_shape.push_back(block_shape[i]);
    block_shape_prod *= block_shape[i];
  }
  axis.push_back(Integer(r_shape.size()));  // axis of (batch / block_shape_prod)
  r_shape.push_back(batch / block_shape_prod);

  for (size_t i = 1; i < num_input_dims; i++) {
    axis.push_back(Integer(r_shape.size()));  // axis of in_shape[i]
    if (axis.size() < (num_block_dims + num_input_dims)) {
      axis.push_back(Integer(r_shape.size() - (num_block_dims + 1)));  // axis of block_shape[i]
    }
    r_shape.push_back(in_shape[i]);
  }

  Array<PrimExpr> r_p_shape;
  r_p_shape.push_back(batch / block_shape_prod);
  for (size_t i = 1; i <= num_block_dims; i++) {
    r_p_shape.push_back(in_shape[i] * block_shape[i - 1]);
  }
  for (size_t i = num_block_dims + 1; i < num_input_dims; i++) {
    r_p_shape.push_back(in_shape[i]);
  }

  tvm::te::Tensor out;
  out = reshape(data, r_shape);
  out = transpose(out, axis);
  out = reshape(out, r_p_shape);

  // Crop the start and end of dimensions of out
  Array<Integer> begin_idx, end_idx, strides;
  for (size_t i = 0; i < r_p_shape.size(); ++i) {
    strides.push_back(Integer(1));
    if (i > 0 && i <= num_block_dims) {
      // prepare begin and end index for spatial dimensions
      int begin_i = static_cast<int>(GetConstInt(crop_begin_list[i - 1]));
      int end_i = static_cast<int>(GetConstInt(crop_end_list[i - 1]));
      int out_i = static_cast<int>(GetConstInt(r_p_shape[i]));
      CHECK_GT(out_i, (begin_i + end_i))
          << "Incorrect crop sizes for (" << i << ")th dim, can not crop more than"
          << " output size" << out_i << " vs " << (begin_i + end_i);
      begin_idx.push_back(begin_i);
      end_idx.push_back(out_i - end_i);
    } else {
      // ignore the batch and remaining dimension
      begin_idx.push_back(Integer(0));
      end_idx.push_back(static_cast<int>(GetConstInt(r_p_shape[i])));
    }
  }

  out = strided_slice(out, begin_idx, end_idx, strides);
  return out;
}

/*!
 * \brief Negative log likelihood loss.
 *
 * \param predictions The prediction tensor.
 * \param targets The target tensor.
 * \param weights A manual rescaling weight given to each class.
 * \param reduction The reduction method to apply to the output.
 * \param ignore_index The target value to ignore.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return The negative log likelihood loss of the predictions and targets.
 */
inline Tensor nll_loss(const Tensor& predictions, const Tensor& targets, const Tensor& weights,
                       std::string reduction = "mean", int ignore_index = -100,
                       const std::string name = "nll_loss", const std::string tag = kBroadcast) {
  if (predictions.ndim() == 1) {
    // corner case: no batch in shape
    // prediction->shape = (C,), targets->shape = (), weights->shape = (C,)
    auto T = tvm::te::compute(
        {},
        [&](const tvm::Array<tvm::tir::Var>& target_indices) {
          auto c = targets();
          return tvm::tir::Select(c != ignore_index, -predictions(c) * weights(c),
                                  tvm::tir::make_const(predictions->dtype, 0));
        },
        name, tag);
    if (reduction == "mean") {
      auto W = tvm::te::compute(
          {},
          [&](const tvm::Array<tvm::tir::Var>& target_indices) {
            auto c = targets();
            return tvm::tir::Select(c != ignore_index, weights(c),
                                    tvm::tir::make_const(predictions->dtype, 0));
          },
          name, tag);
      return topi::divide(T, W);
    } else {
      return T;
    }
  }
  auto T = tvm::te::compute(
      targets->shape,
      [&](const tvm::Array<tvm::tir::Var>& target_indices) {
        auto c = targets(target_indices);
        tvm::Array<tvm::PrimExpr> pred_indices;
        pred_indices.push_back(target_indices[0]);  // batch index
        pred_indices.push_back(c);                  // class index
        for (size_t i = 1; i < target_indices.size(); i++) {
          pred_indices.push_back(target_indices[i]);  // indices for multidimensional loss
        }
        return tvm::tir::Select(c != ignore_index, -predictions(pred_indices) * weights(c),
                                tvm::tir::make_const(predictions->dtype, 0));
      },
      name, tag);
  ICHECK(T->shape.size() != 0);
  if (reduction == "mean") {
    auto W = tvm::te::compute(
        targets->shape,
        [&](const tvm::Array<tvm::tir::Var>& target_indices) {
          auto c = targets(target_indices);
          return tvm::tir::Select(c != ignore_index, weights(c),
                                  tvm::tir::make_const(predictions->dtype, 0));
        },
        name, tag);
    return topi::divide(topi::sum(T, tvm::Array<Integer>(nullptr)),
                        topi::sum(W, tvm::Array<Integer>(nullptr)));
  } else if (reduction == "sum") {
    return topi::sum(T, tvm::Array<Integer>(nullptr));
  } else {  // reduction == "none"
    return T;
  }
}

}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_H_
