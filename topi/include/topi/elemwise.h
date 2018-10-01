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
#include "tvm/ir.h"
#include "tvm/ir_pass.h"

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
TOPI_DECLARE_UNARY_OP(abs);

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

/*!
* \brief Creates an operation that sum each element of a tensor
*
* \param xs The input tensor array
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the sum operation
*/
inline Tensor elemwise_sum(const Array<Tensor>& xs,
                           std::string name = "tensor",
                           std::string tag = kElementWise) {
  CHECK_GT(xs.size(), 0) << "elemwise sum must have at least one input tensor.";
  return compute(xs[0]->shape, [&](const Array<Var>& i) {
      auto sum_expr = xs[0](i);
      for (size_t j = 1; j < xs.size(); j++) {
        sum_expr = sum_expr + xs[j](i);
      }
      return sum_expr;
  }, name, tag);
}

/*!
* \brief Creates an operation that fill a tensor with fill_value
*
* \param shape The shape of a tensor
* \param dtype The Type of fill_value
* \param fill_value The value to be filled
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the full operation
*/
inline Tensor full(const Array<Expr>& shape,
                   Type dtype,
                   const Expr fill_value,
                   std::string name = "tensor",
                   std::string tag = kElementWise) {
  Expr ev = cast(dtype, fill_value);
  if (!ev.defined()) {
    LOG(ERROR) << "Can't cast fill_value to " << dtype;
  }
  return compute(shape, [&](const Array<Var>& i) {
      return ev;
  }, name, tag);
}

/*!
* \brief Creates an operation that construct a tensor with same shape as input tensor,
* then fill a tensor with fill_value
*
* \param x The input tensor
* \param fill_value The value to be filled
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op memeber is the full_like operation
*/
inline Tensor full_like(const Tensor& x,
                        const Expr fill_value,
                        std::string name = "tensor",
                        std::string tag = kElementWise) {
  Expr ev = cast(x->dtype, fill_value);
  return compute(x->shape, [&](const Array<Var>& i) {
      return ev;
  }, name, tag);
}

}  // namespace topi
#endif  // TOPI_ELEMWISE_H_
