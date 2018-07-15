/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Broadcast op constructions
 * \file topi/broadcast.h
 */
#ifndef TOPI_BROADCAST_H_
#define TOPI_BROADCAST_H_

#include <string>
#include <algorithm>
#include "topi/detail/broadcast.h"
#include "topi/detail/constant_utils.h"
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
    CHECK(topi::detail::EqualCheck(output_shape[i], bh.common_shape[i]));
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

#define TOPI_DEFINE_BCAST_OP(Name, ComputeRule)                   \
  inline tvm::Expr Name(const tvm::Expr& a,                       \
                        const tvm::Expr& b) {                     \
    ComputeRule;                                                  \
  }                                                               \
  inline tvm::Tensor Name(const tvm::Tensor& A,                   \
                          const tvm::Tensor& B,                   \
                          std::string name = "tensor",            \
                          std::string tag = kBroadcast) {         \
    auto l = [](tvm::Expr a, tvm::Expr b) { ComputeRule; };       \
    return detail::WithBroadcast(l, A, B, name, tag);             \
  }                                                               \
  inline tvm::Tensor Name(const tvm::Tensor& A,                   \
                          const tvm::Expr& B,                     \
                          std::string name = "tensor",            \
                          std::string tag = kElementWise) {       \
    auto l = [](tvm::Expr a, tvm::Expr b) { ComputeRule; };           \
    return compute(A->shape, [&](const ::tvm::Array<::tvm::Var>& i) { \
        return l(A(i), B);                                        \
      }, name, tag);                                              \
  }                                                               \
  inline tvm::Tensor Name(const tvm::Expr& A,                     \
                          const tvm::Tensor& B,                   \
                          std::string name = "tensor",            \
                          std::string tag = kElementWise) {       \
    auto l = [&](tvm::Expr a, tvm::Expr b) { ComputeRule; };      \
    return compute(B->shape, [&](const ::tvm::Array<::tvm::Var>& i) { \
        return l(A, B(i));                                        \
      }, name, tag);                                              \
  }


#define TOPI_DEFINE_OP_OVERLOAD(Name, OpName)                       \
  inline tvm::Tensor Name(const tvm::Tensor& A,                     \
                          const tvm::Tensor& B) {                   \
    return topi::OpName(A, B);                                      \
  }                                                                 \
  inline tvm::Tensor Name(const tvm::Expr& A,                       \
                          const tvm::Tensor& B) {                   \
    return topi::OpName(A, B);                                      \
  }                                                                 \
  inline tvm::Tensor Name(const tvm::Tensor& A,                     \
                          const tvm::Expr& B) {                     \
    return topi::OpName(A, B);                                      \
  }


/*!
 * \fn add
 * \brief Compute A + B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(add, { return a + b; });
TOPI_DEFINE_OP_OVERLOAD(operator+, add);

/*!
 * \fn subtract
 * \brief Compute A - B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(subtract, { return a - b; });
TOPI_DEFINE_OP_OVERLOAD(operator-, subtract);

/*!
 * \fn multiply
 * \brief Compute A * B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(multiply, { return a * b; });
TOPI_DEFINE_OP_OVERLOAD(operator*, multiply);

/*!
 * \fn divide
 * \brief Compute A / B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(divide, { return a / b; });
TOPI_DEFINE_OP_OVERLOAD(operator/, divide);

/*!
 * \fn mod
 * \brief Compute A % B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(mod, { return a % b; });
TOPI_DEFINE_OP_OVERLOAD(operator%, mod);

/*!
 * \fn maximum
 * \brief Compute maximum(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(maximum, { return tvm::max(a, b); });

/*!
 * \fn minimum
 * \brief Compute minimum(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(minimum, { return tvm::min(a, b); });

/*!
 * \fn power
 * \brief Compute power(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(power, { return tvm::pow(a, b); });

/*!
 * \fn left_shift
 * \brief Compute A << B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(left_shift, { return a << b; });
TOPI_DEFINE_OP_OVERLOAD(operator<<, left_shift);

/*!
 * \fn right_shift
 * \brief Compute A >> B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(right_shift, { return a >> b; });
TOPI_DEFINE_OP_OVERLOAD(operator>>, right_shift);

/*!
 * \fn greater
 * \brief Compute (A > B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(greater, { return (a > b); });

/*!
 * \fn less
 * \brief Compute (A < B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(less, { return (a < b); });

/*!
 * \fn equal
 * \brief Compute (A == B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(equal, { return (a == b); });

/*!
 * \fn not_equal
 * \brief Compute (A != B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(not_equal, { return (a != b); });

/*!
 * \fn greater_equal
 * \brief Compute (A >= B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(greater_equal, { return (a >= b); });

/*!
 * \fn less_equal
 * \brief Compute (A <= B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(less_equal, { return (a <= b); });

}  // namespace topi

#endif  // TOPI_BROADCAST_H_
