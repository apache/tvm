#ifndef TOPI_BROADCAST_H
#define TOPI_BROADCAST_H

#include <topi/detail/broadcast.h>

namespace topi {

inline tvm::Tensor broadcast_to(const tvm::Array<tvm::Expr>& outputShape,
                                const tvm::Tensor& I) {
  CHECK_GE(outputShape.size(), I->shape.size()) <<
    "Not a broadcast, output dimensionality smaller than input.\noutput: " <<
    outputShape << "\nvs\ninput: " << I;
  auto bh = detail::broadcastShape(outputShape, I->shape);
  CHECK_EQ(outputShape.size(), bh.commonShape.size());
  for (int i = 0; i < outputShape.size(); ++i) {
    CHECK(tvm::ir::Equal(outputShape[i], bh.commonShape[i]));
  }
  auto l = [&](tvm::Array<tvm::Var> ovars) {
    return I(detail::inputShapeFromBroadcast(ovars, I, bh.vars2, bh.allVars));
  };
  return tvm::compute(
    tvm::Array<tvm::Expr>(bh.commonShape.begin(), bh.commonShape.end()), l);
}

inline tvm::Tensor broadcast_add(
    const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) {
    return a + b;
  };
  return detail::withBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_sub(
    const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) {
    return a - b;
  };
  return detail::withBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_mul(
    const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) {
    return a * b;
  };
  return detail::withBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_div(
    const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) {
    return a / b;
  };
  return detail::withBroadcast(l, A, B);
}

inline tvm::Tensor broadcast_mod(
    const tvm::Tensor& A, const tvm::Tensor& B) {
  auto l = [&](tvm::Expr a, tvm::Expr b) {
    return a % b;
  };
  return detail::withBroadcast(l, A, B);
}

} // ns topi

#endif // TOPI_BROADCAST_H
