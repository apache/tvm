/*!
 *  Copyright (c) 2017 by Contributors
 * \brief NN op constructions
 * \file nn.h
 */
#ifndef TOPI_NN_H_
#define TOPI_NN_H_

#include <algorithm>
#include <string>

#include "topi/tags.h"
#include "tvm/ir.h"
#include "tvm/ir_pass.h"
#include "tvm/tvm.h"

namespace topi {
namespace detail {

template <typename T>
tvm::Expr Map(const tvm::Array<tvm::Expr>& exprs, T op) {
  CHECK_GE(exprs.size(), 1);
  tvm::Expr res = exprs[0];
  for (int i = 1; i < exprs.size(); ++i) {
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
                        std::string name = "tensor",
                        std::string tag = kElementWise) {
  return tvm::compute(
      t->shape,
      [&](const tvm::Array<tvm::Var>& i) { return tvm::max(t(i), threshold); },
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
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the relu operation
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
                       std::string name = "tensor",
                       std::string tag = kElementWise) {
  if (pad_after.size() < pad_before.size()) {
    for (int i = pad_after.size(); i < pad_before.size(); ++i) {
      pad_after.push_back(pad_before[i]);
    }
  }
  CHECK_GE(pad_before.size(), 1);
  CHECK_EQ(pad_before.size(), pad_after.size());
  tvm::Array<tvm::Expr> output_shape;
  for (int i = 0; i < t->shape.size(); ++i) {
    if (i >= pad_before.size()) {
      output_shape.push_back(t->shape[i]);
    } else {
      output_shape.push_back(
          tvm::ir::Simplify(t->shape[i] + pad_before[i] + pad_after[i]));
    }
  }
  auto l = [&](tvm::Array<tvm::Var> ovars) {
    tvm::Array<tvm::Expr> indices;
    tvm::Array<tvm::Expr> sel;
    for (int i = 0; i < t->shape.size(); ++i) {
      if (i >= pad_before.size()) {
        indices.push_back(ovars[i]);
        continue;
      }
      if (!tvm::ir::Equal(pad_before[i], 0)) {
        sel.push_back(ovars[i] >= pad_before[i]);
        indices.push_back(ovars[i] - pad_before[i]);
      } else {
        indices.push_back(ovars[i]);
      }
      if (!tvm::ir::Equal(pad_after[i], 0)) {
        sel.push_back(tvm::ir::Simplify(ovars[i] < pad_before[i] + t->shape[i]));
      }
    }
    return tvm::select(detail::Map(sel, tvm::ir::And::make), t(indices), 0);
  };
  return tvm::compute(output_shape, l, name, tag);
}

/*!
 * \brief Creates an operation that calculates a matrix multiplication
 *  (row-major notation):
 *      A(i, k) * B(k, j), if trans_a == trans_b
 *          the usual transposed combinations, otherwise
 *
 * \param A The matrix A
 * \param B The matrix B
 * \param trans_a Is A's layout transposed?
 * \param trans_b Is B's layout transposed?
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the matmult operation
 */
inline tvm::Tensor matmult(const tvm::Tensor& A,
                           const tvm::Tensor& B,
                           bool trans_a = false,
                           bool trans_b = false,
                           std::string name = "tensor",
                           std::string tag = kMatMult) {
  tvm::Array<tvm::Expr> output_shape{A->shape[trans_a ? 1 : 0],
                                     B->shape[trans_b ? 0 : 1]};
  auto k = tvm::reduce_axis(tvm::Range{0, A->shape[trans_a ? 0 : 1]}, "k");
  auto l = [&](tvm::Var i, tvm::Var j) {
    return tvm::sum((trans_a ? A[k][i] : A[i][k]) * (trans_b ? B[j][k] : B[k][j]),
                    {k});
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
                               std::string name = "tensor",
                               std::string tag = kConv2dNCHW) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
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
                               std::string name = "tensor",
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
                                         std::string name = "tensor",
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
                                         std::string name = "tensor",
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
                                      std::string name = "tensor",
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
