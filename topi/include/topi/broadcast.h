/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Broadcast op constructions
 * \file topi/broadcast.h
 */
#ifndef TOPI_BROADCAST_H_
#define TOPI_BROADCAST_H_

#include <string>

#include "topi/detail/broadcast.h"
#include "topi/tags.h"

namespace topi {

/*!
 * \brief Creates an operation that broadcasts a tensor into a compatible
 * shape according to numpy's rules
 *
 * \param t The input tensor
 * \param output_shape The target output shape, must be compatible
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a broadcast operation
 */
inline tvm::Tensor broadcast_to(const tvm::Tensor& t,
                                const tvm::Array<tvm::Expr>& output_shape,
                                std::string name = "tensor",
                                std::string tag = kBroadcast) {
  CHECK_GE(output_shape.size(), t->shape.size())
      << "Not a broadcast, output dimensionality smaller than input.\noutput: "
      << output_shape << "\nvs\ninput: " << t;
  auto bh = detail::BroadcastShape(output_shape, t->shape);
  CHECK_EQ(output_shape.size(), bh.common_shape.size());
  for (size_t i = 0; i < output_shape.size(); ++i) {
    CHECK(tvm::ir::Equal(output_shape[i], bh.common_shape[i]));
  }
  auto l = [&](tvm::Array<tvm::Var> ovars) {
    return t(detail::InputIndexFromBroadcast(ovars, t, bh.vars2, bh.all_vars));
  };
  return tvm::compute(
      tvm::Array<tvm::Expr>(bh.common_shape.begin(), bh.common_shape.end()),
      l,
      name,
      tag);
}

/*!
 * \brief Creates an operation that performs pointwise addition of 2 tensors
 * and broadcasts them into a common compatible shape where necessary,
 * according to numpy's rules
 *
 * \param A The first tensor to add
 * \param B The second tensor to add
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a pointwise addition with broadcast
 */
inline tvm::Tensor broadcast_add(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a + b; };
  return detail::WithBroadcast(l, A, B, name, tag);
}

/*!
 * \brief Creates an operation that performs pointwise subtraction of 2 tensors
 * and broadcasts them into a common compatible shape where necessary,
 * according to numpy's rules
 *
 * \param A The first tensor
 * \param B The second tensor to subtract from the first
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a pointwise subtraction with broadcast
 */
inline tvm::Tensor broadcast_sub(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a - b; };
  return detail::WithBroadcast(l, A, B, name, tag);
}

/*!
 * \brief Creates an operation that performs pointwise multiplication of 2
 * tensors and broadcasts them into a common compatible shape where necessary,
 * according to numpy's rules
 *
 * \param A The first tensor to multiply
 * \param B The second tensor to multiply
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a pointwise multiplication with broadcast
 */
inline tvm::Tensor broadcast_mul(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a * b; };
  return detail::WithBroadcast(l, A, B, name, tag);
}

/*!
 * \brief Creates an operation that performs pointwise division of 2 tensors
 * and broadcasts them into a common compatible shape where necessary,
 * according to numpy's rules
 *
 * \param A The first tensor
 * \param B The second tensor to divide the first tensor with
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a pointwise division with broadcast
 */
inline tvm::Tensor broadcast_div(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a / b; };
  return detail::WithBroadcast(l, A, B, name, tag);
}

/*!
 * \brief Creates an operation that performs pointwise modulo remainder of 2
 * tensors and broadcasts them into a common compatible shape where necessary,
 * according to numpy's rules
 *
 * \param A The first tensor
 * \param B The second tensor to compute A % B
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a pointwise modulo remainder with
 * broadcast
 */
inline tvm::Tensor broadcast_mod(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return a % b; };
  return detail::WithBroadcast(l, A, B, name, tag);
}

/*!
* \brief Creates an operation that performs pointwise maximum of 2 tensors
* and broadcasts them into a common compatible shape where necessary,
* according to numpy's rules
*
* \param A The first tensor
* \param B The second tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is a pointwise maximum with broadcast
*/
inline tvm::Tensor broadcast_maximum(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return tvm::max(a, b); };  // NOLINT(*)
  return detail::WithBroadcast(l, A, B, name, tag);
}

/*!
* \brief Creates an operation that performs pointwise minimum of 2 tensors
* and broadcasts them into a common compatible shape where necessary,
* according to numpy's rules
*
* \param A The first tensor
* \param B The second tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is a pointwise minimum with broadcast
*/
inline tvm::Tensor broadcast_minimum(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return tvm::min(a, b); };  // NOLINT(*)
  return detail::WithBroadcast(l, A, B, name, tag);
}

/*!
* \brief Creates an operation that raises one tensor to the power of another
* pointwise and broadcasts them into a common compatible shape where necessary,
* according to numpy's rules
*
* \param A The first tensor
* \param B The second tensor to compute pow(A, B)
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is a pointwise pow with
* broadcast
*/
inline tvm::Tensor broadcast_pow(const tvm::Tensor& A,
                                 const tvm::Tensor& B,
                                 std::string name = "tensor",
                                 std::string tag = kBroadcast) {
  auto l = [&](tvm::Expr a, tvm::Expr b) { return tvm::pow(a, b); };
  return detail::WithBroadcast(l, A, B, name, tag);
}

}  // namespace topi

#endif  // TOPI_BROADCAST_H_
