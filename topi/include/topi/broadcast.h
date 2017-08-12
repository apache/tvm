/*
 *  Copyright (c) 2017 by Contributors
 * \brief Broadcast op constructions
 * \file broadcast.h
 */
#ifndef TOPI_BROADCAST_H_
#define TOPI_BROADCAST_H_

#include <topi/detail/broadcast.h>

namespace topi {

inline tvm::Tensor broadcast_to(const tvm::Tensor& I,
                                const tvm::Array<tvm::Expr>& output_shape) {
  CHECK_GE(output_shape.size(), I->shape.size())
      << "Not a broadcast, output dimensionality smaller than input.\noutput: "
      << output_shape << "\nvs\ninput: " << I;
  auto bh = detail::BroadcastShape(output_shape, I->shape);
  CHECK_EQ(output_shape.size(), bh.common_shape.size());
  for (int i = 0; i < output_shape.size(); ++i) {
    CHECK(tvm::ir::Equal(output_shape[i], bh.common_shape[i]));
  }
  auto l = [&](tvm::Array<tvm::Var> ovars) {
    return I(detail::InputIndexFromBroadcast(ovars, I, bh.vars2, bh.all_vars));
  };
  return tvm::compute(
      tvm::Array<tvm::Expr>(bh.common_shape.begin(), bh.common_shape.end()), l);
}

inline tvm::Tensor broadcast_add(const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a + b; };
  return detail::WithBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_sub(const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a - b; };
  return detail::WithBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_mul(const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a * b; };
  return detail::WithBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_div(const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a / b; };
  return detail::WithBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_mod(const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a % b; };
  return detail::WithBroadcast(l, A, B);
}

}  // namespace topi

#endif  // TOPI_BROADCAST_H_
