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
 * \file elemwise.h
 * \brief Elementwise op constructions
 */
#ifndef TOPI_ELEMWISE_H_
#define TOPI_ELEMWISE_H_

#include <string>

#include "topi/tags.h"
#include "tvm/ir.h"
#include "tvm/ir_pass.h"
#include "broadcast.h"

namespace topi {
using namespace tvm;

// Unary intrinsic operators
#define TOPI_DECLARE_UNARY_OP(OpName)                           \
  inline Tensor OpName(const Tensor& x,                         \
                       std::string name = "T_" #OpName,         \
                       std::string tag = kElementWise) {        \
    return compute(x->shape, [&](const Array<Var>& i) {         \
        return ::tvm::OpName(x(i));                             \
      }, name, tag);                                            \
  }

TOPI_DECLARE_UNARY_OP(exp);
TOPI_DECLARE_UNARY_OP(sigmoid);
TOPI_DECLARE_UNARY_OP(sqrt);
TOPI_DECLARE_UNARY_OP(log);
TOPI_DECLARE_UNARY_OP(floor);
TOPI_DECLARE_UNARY_OP(ceil);
TOPI_DECLARE_UNARY_OP(round);
TOPI_DECLARE_UNARY_OP(trunc);
TOPI_DECLARE_UNARY_OP(abs);

/*
 * \brief Fast_tanh_float implementation from Eigen
 * https://github.com/eigenteam/eigen-git-mirror/blob/master/Eigen/src/Core/MathFunctionsImpl.h#L26
 */
inline Tensor fast_tanh_float(const Tensor& in,
                              std::string name,
                              std::string tag) {
  // Clamp the inputs to the range [-9, 9] since anything outside
  // this range is +/-1.0f in single-precision.
  auto x = maximum(minimum(in, make_const(in->dtype, 9.0)), make_const(in->dtype, -9.0));

  // The monomial coefficients of the numerator polynomial (odd).
  auto alpha_1 = make_const(in->dtype, 4.89352455891786e-03);
  auto alpha_3 = make_const(in->dtype, 6.37261928875436e-04);
  auto alpha_5 = make_const(in->dtype, 1.48572235717979e-05);
  auto alpha_7 = make_const(in->dtype, 5.12229709037114e-08);
  auto alpha_9 = make_const(in->dtype, -8.60467152213735e-11);
  auto alpha_11 = make_const(in->dtype, 2.00018790482477e-13);
  auto alpha_13 = make_const(in->dtype, -2.76076847742355e-16);

  // The monomial coefficients of the denominator polynomial (even).
  auto beta_0 = make_const(in->dtype, 4.89352518554385e-03);
  auto beta_2 = make_const(in->dtype, 2.26843463243900e-03);
  auto beta_4 = make_const(in->dtype, 1.18534705686654e-04);
  auto beta_6 = make_const(in->dtype, 1.19825839466702e-06);

  return compute(x->shape,
                 [&](const Array<Var>& i) {
                   auto x2 = x(i) * x(i);
                   auto p = x2 * alpha_13 + alpha_11;
                   p = x2 * p + alpha_9;
                   p = x2 * p + alpha_7;
                   p = x2 * p + alpha_5;
                   p = x2 * p + alpha_3;
                   p = x2 * p + alpha_1;
                   p = x(i) * p;

                   auto q = x2 * beta_6 + beta_4;
                   q = x2 * q + beta_2;
                   q = x2 * q + beta_0;
                   return p / q;
                 },
                 name, tag);
}

/*!
* \brief Creates an operation that returns hyperbolic tanh of a given tensor
*
* \param x The input tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is tanh
*/
inline Tensor tanh(const Tensor& x,
                   std::string name = "T_tanh",
                   std::string tag = kElementWise) {
  if (x->dtype == Float(32)) {
    // invoke fast_tanh_float implementation
    return fast_tanh_float(x, name, tag);
  } else {
    // fallback to default implementation
    return compute(x->shape, [&](const Array<Var>& i) {
      return ::tvm::tanh(x(i));
    }, name, tag);
  }
}

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
                       std::string name = "T_identity",
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
                       std::string name = "T_negative",
                       std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    return -x(i);
  }, name, tag);
}

/*!
* \brief Creates an operation that returns the logical NOT of a given tensor
*
* \param x The input tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the logical NOT operation
*/
inline Tensor logical_not(const Tensor& x,
                          std::string name = "T_logical_not",
                          std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    return !x(i);
  }, name, tag);
}

/*!
* \brief Returns the sign of the tensor
*
* \param x The input tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the sign
*/
inline Tensor sign(const Tensor& x,
                   std::string name = "T_sign",
                   std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    Expr zero = make_zero(x->dtype);
    Expr one = make_const(x->dtype, 1);
    Expr minus_one = make_const(x->dtype, -1);
    auto s1 = tvm::ir::Select::make((x(i) < zero), minus_one, zero);
    auto s2 = tvm::ir::Select::make((x(i) > zero), one, s1);
    return s2;
  }, name, tag);
}

/*!
* \brief Creates an operation that returns rsqrt of a given tensor
*
* \param x The input tensor
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return A Tensor whose op member is the rsqrt operation
*/
inline Tensor rsqrt(const Tensor& x,
                       std::string name = "tensor",
                       std::string tag = kElementWise) {
  return compute(x->shape, [&](const Array<Var>& i) {
    Expr one = make_const(x->dtype, 1);
    return one/tvm::sqrt(x(i));
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
                   std::string name = "T_clip",
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
                   std::string name = "T_cast",
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
                           std::string name = "T_elemwise_sum",
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
                   std::string name = "T_full",
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
                        std::string name = "T_full_like",
                        std::string tag = kElementWise) {
  Expr ev = cast(x->dtype, fill_value);
  return compute(x->shape, [&](const Array<Var>& i) {
      return ev;
  }, name, tag);
}

}  // namespace topi
#endif  // TOPI_ELEMWISE_H_
