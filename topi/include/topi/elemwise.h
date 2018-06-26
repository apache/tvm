/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise.h
 * \brief Elementwise op constructions
 */
#ifndef TOPI_ELEMWISE_H_
#define TOPI_ELEMWISE_H_

#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

// Unary intrinsic operators
#define TOPI_DECLARE_UNARY_OP(OpName)                           \
  inline Tensor OpName(const Tensor& x,                         \
                       std::string name = "tensor",             \
                       std::string tag = kElementWise) {        \
    return compute(x->shape, [&](const Array<Var>& i) {         \
        return ::tvm::OpName(x(i));                             \
      }, name, tag);                                            \
  }

TOPI_DECLARE_UNARY_OP(exp);
TOPI_DECLARE_UNARY_OP(tanh);
TOPI_DECLARE_UNARY_OP(sigmoid);
TOPI_DECLARE_UNARY_OP(sqrt);
TOPI_DECLARE_UNARY_OP(log);
TOPI_DECLARE_UNARY_OP(floor);
TOPI_DECLARE_UNARY_OP(ceil);
TOPI_DECLARE_UNARY_OP(round);
TOPI_DECLARE_UNARY_OP(trunc);

/*!
* \brief Creates an operation that returns identity of a given tensor
*
* \param x The input tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the identity operation
*/
inline Tensor identity(const Tensor& x,
                       std::string name = "tensor",
                       std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    return x(i);
  }, name, tag);
}

/*!
* \brief Creates an operation that returns the negation of a given tensor
*
* \param x The input tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the negation operation
*/
inline Tensor negative(const Tensor& x,
                       std::string name = "tensor",
                       std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    return -x(i);
  }, name, tag);
}

/*!
* \brief Creates an operation that clips each element of a tensor to
* the interval [a_min, a_max]
*
* \param x The input tensor
* \param a_min The inclusive lower bound of the interval
* \param a_max The inclusive upper bound of the interval
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the clip operation
*/
inline Tensor clip(const Tensor& x,
                   const Expr& a_min,
                   const Expr& a_max,
                   std::string name = "tensor",
                   std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    auto min_val = tvm::cast(x->dtype, a_min);
    auto max_val = tvm::cast(x->dtype, a_max);
    return tvm::max(tvm::min(x(i), max_val), min_val);  // NOLINT(*)
  }, name, tag);
}

/*!
 * \brief Cast each element of x to the given type. If expr is
 * scalar and type is a corresponding vector type, a
 * Broadcast is generated, otherwise a Cast is generated.
 *
 * \param x The input tensor
 * \param type The type to cast to
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the cast operation
 */
inline Tensor cast(const Tensor& x,
                   Type type,
                   std::string name = "tensor",
                   std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    auto expr = x(i);
    if (expr.type().code() == type.code() && expr.type().bits() == type.bits()) {
      if (expr.type().lanes() == type.lanes()) {
        return expr;
      } else if (expr.type().lanes() == 1 && type.lanes() > 1) {
        return tvm::ir::Broadcast::make(expr, type.lanes());
      }
    }

    return tvm::cast(type, x(i));
  }, name, tag);
}

}  // namespace topi
#endif  // TOPI_ELEMWISE_H_
