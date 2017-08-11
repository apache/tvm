/*
 *  Copyright (c) 2017 by Contributors
 * \file topi.h
 * \brief Elementwise op constructions
 */
#ifndef TOPI_NN_H_
#define TOPI_NN_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/tvm.h>

namespace topi {

template <typename T>
tvm::Expr map(const tvm::Array<tvm::Expr>& exprs, T op) {
  CHECK_GE(exprs.size(), 1);
  tvm::Expr res = exprs[0];
  for (int i = 1; i < exprs.size(); ++i) {
    res = op(res, exprs[i]);
  }
  return res;
}

template <typename T>
inline tvm::Tensor relu(const tvm::Tensor& x, T threshold) {
  return tvm::compute(x->shape, [&](const tvm::Array<tvm::Var>& i) {
    return tvm::max(x(i), threshold);
    }, "tensor", "ewise");
}

inline tvm::Tensor pad(
    const tvm::Tensor& t,
    const tvm::Array<tvm::Expr>& padBefore,
    tvm::Array<tvm::Expr> padAfter = tvm::Array<tvm::Expr>()) {
  if (padAfter.size() < padBefore.size()) {
    for(int i = padAfter.size(); i < padBefore.size(); ++i) {
      padAfter.push_back(padBefore[i]);
    }
  }
  CHECK_GE(padBefore.size(), 1);
  CHECK_EQ(padBefore.size(), padAfter.size());
  tvm::Array<tvm::Expr> outputShape;
  for (int i = 0; i < t->shape.size(); ++i) {
    if (i >= padBefore.size()) {
      outputShape.push_back(t->shape[i]);
    } else {
      outputShape.push_back(
        tvm::ir::Simplify(t->shape[i] + padBefore[i] + padAfter[i]));
    }
  }
  auto l = [&](tvm::Array<tvm::Var> ovars) {
    tvm::Array<tvm::Expr> indices;
    tvm::Array<tvm::Expr> sel;
    for (int i = 0; i < t->shape.size(); ++i) {
      if (i >= padBefore.size()) {
        indices.push_back(ovars[i]);
        continue;
      }
      if (!tvm::ir::Equal(padBefore[i], 0)) {
        sel.push_back(ovars[i] >= padBefore[i]);
        indices.push_back(ovars[i] - padBefore[i]);
      } else {
        indices.push_back(ovars[i]);
      }
      if (!tvm::ir::Equal(padAfter[i], 0)) {
        sel.push_back(tvm::ir::Simplify(ovars[i] < padBefore[i] + t->shape[i]));
      }
    }
    return tvm::select(map(sel, tvm::ir::And::make), t(indices), 0);
  };
  return tvm::compute(outputShape, l, "tensor", "ewise");
}

// Returns a compute that calculates a row-major matrix multiplication:
//   A(i, k) * B(k, j), if transA == transB
//   the usual transposed combinations, otherwise
inline tvm::Tensor matmult(const tvm::Tensor& A,
                           const tvm::Tensor& B,
                           bool transA = false,
                           bool transB = false) {
  tvm::Array<tvm::Expr> outputShape{
    A->shape[transA ? 1 : 0],
    B->shape[transB ? 0 : 1]
  };
  auto k = tvm::reduce_axis(tvm::Range{0, A->shape[transA ? 0 : 1]}, "k");
  auto l = [&](tvm::Var i, tvm::Var j) {
    return tvm::sum(
      (transA ? A[k][i] : A[i][k]) * (transB ? B[j][k] : B[k][j]),
      {k});
  };
  return tvm::compute(outputShape, l);
}

inline tvm::Tensor conv2d_nchw(const tvm::Tensor& I,
                               const tvm::Tensor& W,
                               int padH = 0,
                               int padW = 0,
                               int strideH = 1,
                               int strideW = 1) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::Expr> outputShape{
    I->shape[0],                   // B
    W->shape[1],                   // O
    (I->shape[2] - W->shape[2] + 2 * padH) / strideH + 1, // H
    (I->shape[3] - W->shape[3] + 2 * padW) / strideW + 1  // W
  };
  auto i  = tvm::reduce_axis(tvm::Range{0, I->shape[1]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[2]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[3]}, "kw");
  auto T = (padH == 0 && padW == 0) ?
    I : pad(I, {tvm::Expr(0), tvm::Expr(0), padH, padW});
  auto l = [&](tvm::Var b, tvm::Var o, tvm::Var h, tvm::Var w) {
    return tvm::sum(
      T(b, i, strideH * h + kh, strideW * w + kw) *  W(i, o, kh, kw),
      {i, kh, kw}
    );
  };
  return tvm::compute(outputShape, l);
}

inline tvm::Tensor conv2d_hwcn(const tvm::Tensor& I,
                               const tvm::Tensor& W,
                               int padH = 0,
                               int padW = 0,
                               int strideH = 1,
                               int strideW = 1) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::Expr> outputShape{
    (I->shape[2] - W->shape[2] + 2 * padH) / strideH + 1, // H
    (I->shape[3] - W->shape[3] + 2 * padW) / strideW + 1, // W
    I->shape[2],                   // B
    W->shape[3]                    // O
  };
  auto i  = tvm::reduce_axis(tvm::Range{0, I->shape[3]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[0]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[1]}, "kw");
  auto T = (padH == 0 && padW == 0) ? I : pad(I, {padH, padW});
  auto l = [&](tvm::Var b, tvm::Var o, tvm::Var h, tvm::Var w) {
    return tvm::sum(
      T(strideH * h + kh, strideW * w + kw, i, b) * W(kh, kw, i, o),
      {i, kh, kw}
    );
  };
  return tvm::compute(outputShape, l);
}

inline tvm::Tensor depthwise_conv2d_nchw(const tvm::Tensor& I,
                                         const tvm::Tensor& W,
                                         int padH = 0,
                                         int padW = 0,
                                         int strideH = 1,
                                         int strideW = 1) {
  CHECK_EQ(4, I->shape.size());
  CHECK_EQ(4, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  auto pCM = W->shape[1]; // channel_multiplier
  tvm::Array<tvm::Expr> outputShape{
    I->shape[0],                   // B
    W->shape[1],                   // O
    (I->shape[2] - W->shape[2] + 2 * padH) / strideH + 1, // H
    (I->shape[3] - W->shape[3] + 2 * padW) / strideW + 1  // W
  };
  auto i  = tvm::reduce_axis(tvm::Range{0, I->shape[1]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[2]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[3]}, "kw");
  auto T = (padH == 0 && padW == 0) ?
    I : pad(I, {tvm::Expr(0), tvm::Expr(0), padH, padW});
  auto l = [&](tvm::Var b, tvm::Var o, tvm::Var h, tvm::Var w) {
    return tvm::sum(
      T(b, i / pCM, strideH * h + kh, strideW * w + kw) *  W(i / pCM, o % pCM, kh, kw),
      {i, kh, kw}
    );
  };
  return tvm::compute(outputShape, l);
}

inline tvm::Tensor group_conv2d_ngchw(const tvm::Tensor& I,
                                      const tvm::Tensor& W,
                                      int padH = 0,
                                      int padW = 0,
                                      int strideH = 1,
                                      int strideW = 1) {
  CHECK_EQ(5, I->shape.size());
  CHECK_EQ(5, W->shape.size());
  auto pH = I->shape[2];
  auto pW = I->shape[3];
  tvm::Array<tvm::Expr> outputShape{
    I->shape[0],                   // B
    I->shape[1],                   // G
    W->shape[2],                   // O
    (I->shape[3] - W->shape[3] + 2 * padH) / strideH + 1, // H
    (I->shape[4] - W->shape[4] + 2 * padW) / strideW + 1  // W
  };
  auto i  = tvm::reduce_axis(tvm::Range{0, I->shape[2]}, "i");
  auto kh = tvm::reduce_axis(tvm::Range{0, W->shape[3]}, "kh");
  auto kw = tvm::reduce_axis(tvm::Range{0, W->shape[4]}, "kw");

  auto T = (padH == 0 && padW == 0) ?
    I : pad(I, {tvm::Expr(0), tvm::Expr(0), tvm::Expr(0), padH, padW});
  auto l = [&](tvm::Var b, tvm::Var g, tvm::Var o, tvm::Var h, tvm::Var w) {
    return tvm::sum(
      I(b, g, i, strideH * h + kh, strideW * w + kw) * W(g, i, o, kh, kw),
      {i, kh, kw}
    );
  };
  return tvm::compute(outputShape, l);
}

}  // namespace topi
#endif  // TOPI_NN_H_
