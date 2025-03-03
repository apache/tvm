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
 * \file elemwise.h
 * \brief Elementwise op constructions
 */
#ifndef TVM_TOPI_ELEMWISE_H_
#define TVM_TOPI_ELEMWISE_H_

#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <string>

#include "broadcast.h"

namespace tvm {
namespace topi {

using namespace tvm::te;

// Unary intrinsic operators
#define TOPI_DECLARE_UNARY_OP(OpName)                                                   \
  inline Tensor OpName(const Tensor& x, std::string name = "T_" #OpName,                \
                       std::string tag = kElementWise) {                                \
    return compute(                                                                     \
        x->shape, [&](const Array<Var>& i) { return ::tvm::OpName(x(i)); }, name, tag); \
  }

TOPI_DECLARE_UNARY_OP(exp);
TOPI_DECLARE_UNARY_OP(erf);
TOPI_DECLARE_UNARY_OP(sigmoid);
TOPI_DECLARE_UNARY_OP(sqrt);
TOPI_DECLARE_UNARY_OP(log);
TOPI_DECLARE_UNARY_OP(log2);
TOPI_DECLARE_UNARY_OP(log10);
TOPI_DECLARE_UNARY_OP(floor);
TOPI_DECLARE_UNARY_OP(ceil);
TOPI_DECLARE_UNARY_OP(round);
TOPI_DECLARE_UNARY_OP(trunc);
TOPI_DECLARE_UNARY_OP(abs);
TOPI_DECLARE_UNARY_OP(cos);
TOPI_DECLARE_UNARY_OP(cosh);
TOPI_DECLARE_UNARY_OP(tan);
TOPI_DECLARE_UNARY_OP(sin);
TOPI_DECLARE_UNARY_OP(sinh);
TOPI_DECLARE_UNARY_OP(acos);
TOPI_DECLARE_UNARY_OP(acosh);
TOPI_DECLARE_UNARY_OP(asin);
TOPI_DECLARE_UNARY_OP(asinh);
TOPI_DECLARE_UNARY_OP(atan);
TOPI_DECLARE_UNARY_OP(atanh);
TOPI_DECLARE_UNARY_OP(isnan);
TOPI_DECLARE_UNARY_OP(tanh);
TOPI_DECLARE_UNARY_OP(isfinite);
TOPI_DECLARE_UNARY_OP(isinf);

/*!
 * \brief Fast_tanh_float implementation from Eigen
 * https://github.com/eigenteam/eigen-git-mirror/blob/master/Eigen/src/Core/MathFunctionsImpl.h#L26
 */
inline Tensor fast_tanh_float(const Tensor& in, std::string name, std::string tag) {
  // Clamp the inputs to the range [-9, 9] since anything outside
  // this range is +/-1.0f in single-precision.
  auto x = maximum(make_const(in->dtype, -9.0), minimum(make_const(in->dtype, 9.0), in));

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

  return compute(
      x->shape,
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
inline Tensor fast_tanh(const Tensor& x, std::string name = "T_fast_tanh",
                        std::string tag = kElementWise) {
  if (x->dtype == DataType::Float(32)) {
    // invoke fast_tanh_float implementation
    return fast_tanh_float(x, name, tag);
  } else {
    // fallback to default implementation
    return compute(
        x->shape, [&](const Array<Var>& i) { return ::tvm::tanh(x(i)); }, name, tag);
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
inline Tensor identity(const Tensor& x, std::string name = "T_identity",
                       std::string tag = kElementWise) {
  return compute(
      x->shape, [&](const Array<Var>& i) { return x(i); }, name, tag);
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
inline Tensor negative(const Tensor& x, std::string name = "T_negative",
                       std::string tag = kElementWise) {
  return compute(
      x->shape, [&](const Array<Var>& i) { return -x(i); }, name, tag);
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
inline Tensor logical_not(const Tensor& x, std::string name = "T_logical_not",
                          std::string tag = kElementWise) {
  return compute(
      x->shape, [&](const Array<Var>& i) { return !x(i); }, name, tag);
}

/*!
 * \brief Creates an operation that returns the bitwise NOT of a given tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the bitwise NOT operation
 */
inline Tensor bitwise_not(const Tensor& x, std::string name = "T_bitwise_not",
                          std::string tag = kElementWise) {
  return compute(
      x->shape, [&](const Array<Var>& i) { return ~x(i); }, name, tag);
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
inline Tensor sign(const Tensor& x, std::string name = "T_sign", std::string tag = kElementWise) {
  return compute(
      x->shape,
      [&](const Array<Var>& i) {
        PrimExpr zero = make_zero(x->dtype);
        PrimExpr one = make_const(x->dtype, 1);
        PrimExpr minus_one = make_const(x->dtype, -1);
        auto s1 = tvm::tir::Select((x(i) < zero), minus_one, zero);
        auto s2 = tvm::tir::Select((x(i) > zero), one, s1);
        return s2;
      },
      name, tag);
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
inline Tensor rsqrt(const Tensor& x, std::string name = "tensor", std::string tag = kElementWise) {
  return compute(
      x->shape,
      [&](const Array<Var>& i) {
        PrimExpr one = make_const(x->dtype, 1);
        return one / tvm::sqrt(x(i));
      },
      name, tag);
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
inline Tensor clip(const Tensor& x, const PrimExpr& a_min, const PrimExpr& a_max,
                   std::string name = "T_clip", std::string tag = kElementWise) {
  return compute(
      x->shape,
      [&](const Array<Var>& i) {
        auto min_val = tvm::cast(x->dtype, a_min);
        auto max_val = tvm::cast(x->dtype, a_max);
        return tvm::max(tvm::min(x(i), max_val), min_val);  // NOLINT(*)
      },
      name, tag);
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
inline Tensor cast(const Tensor& x, DataType type, std::string name = "T_cast",
                   std::string tag = kElementWise) {
  return compute(
      x->shape,
      [&](const Array<Var>& i) -> PrimExpr {
        auto expr = x(i);
        if (expr.dtype().code() == type.code() && expr.dtype().bits() == type.bits()) {
          if (expr.dtype().lanes() == type.lanes()) {
            return expr;
          } else if (expr.dtype().lanes() == 1 && type.is_vector()) {
            return tvm::tir::Broadcast(expr, type.lanes());
          }
        }

        return tvm::cast(type, x(i));
      },
      name, tag);
}

/*!
 * \brief Reinterpret each element of x to the given type.

 * \param x The input tensor
 * \param type The type to cast to
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the reinterpret operation
 */
inline Tensor reinterpret(const Tensor& x, DataType type, std::string name = "tensor",
                          std::string tag = kElementWise) {
  return compute(
      x->shape, [&](const Array<Var>& i) { return reinterpret(type, x(i)); }, name, tag);
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
inline Tensor elemwise_sum(const Array<Tensor>& xs, std::string name = "T_elemwise_sum",
                           std::string tag = kElementWise) {
  ICHECK_GT(xs.size(), 0) << "elemwise sum must have at least one input tensor.";
  return compute(
      xs[0]->shape,
      [&](const Array<Var>& i) {
        auto sum_expr = xs[0](i);
        for (size_t j = 1; j < xs.size(); j++) {
          sum_expr = sum_expr + xs[j](i);
        }
        return sum_expr;
      },
      name, tag);
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
inline Tensor full(const Array<PrimExpr>& shape, DataType dtype, const PrimExpr fill_value,
                   std::string name = "T_full", std::string tag = kElementWise) {
  PrimExpr ev = cast(dtype, fill_value);
  if (!ev.defined()) {
    LOG(ERROR) << "Can't cast fill_value to " << dtype;
  }
  return compute(
      shape, [&](const Array<Var>& i) { return ev; }, name, tag);
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
inline Tensor full_like(const Tensor& x, const PrimExpr fill_value,
                        std::string name = "T_full_like", std::string tag = kElementWise) {
  PrimExpr ev = cast(x->dtype, fill_value);
  return compute(
      x->shape, [&](const Array<Var>& i) { return ev; }, name, tag);
}

/*!
 * \brief Fast exponential function implementation
 *
 * \param _x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is exponent operation
 *
 * \note Function computes:
 * log2(e^x) = x * log2(e) * log2(2) =>
 * log2(e^x) = log2(2^(x*log2(e))) =>
 * e^x = 2^(x*log2(e))
 * Splitting power x*log2(e) into integer and fractional parts:
 * e^(n+f) = e^n * e^f
 * n = floor(x*log2(e) + 1/2)
 * f = x - n * ln(2)
 * exp(x) = 2^n * exp(y)
 * Approximation for fractional part:
 * y = exp(f) = 1 + 2 * P(x**2)/(Q(x**2) - P(x**2))
 */
inline Tensor fast_exp_float32(const Tensor& _x, std::string name, std::string tag) {
  auto x_hi = make_const(DataType::Float(32), 88.3762626647950f);
  auto x_lo = make_const(DataType::Float(32), -88.3762626647949f);
  auto log2e = make_const(DataType::Float(32), 1.44269504088896341f);
  auto ln2 = make_const(DataType::Float(32), 0.6931471805599453f);
  PrimExpr p[6] = {make_const(DataType::Float(32), 1.9875691500E-4f),
                   make_const(DataType::Float(32), 1.3981999507E-3f),
                   make_const(DataType::Float(32), 8.3334519073E-3f),
                   make_const(DataType::Float(32), 4.1665795894E-2f),
                   make_const(DataType::Float(32), 1.6666665459E-1f),
                   make_const(DataType::Float(32), 5.0000001201E-1f)};
  auto one = make_const(DataType::Float(32), 1.0f);
  auto one_half = make_const(DataType::Float(32), 0.5f);
  auto b = make_const(DataType::Float(32), 127.0f);

  return compute(
      _x->shape,
      [&](const Array<Var>& i) {
        // clamp x
        auto x = ::tvm::max(::tvm::min(_x(i), x_hi), x_lo);
        // integer part
        auto n = ::tvm::floor(x * log2e + one_half);
        // fractional part
        auto f = x - n * ln2;
        auto y =
            (((((p[0] * f + p[1]) * f + p[2]) * f + p[3]) * f + p[4]) * f + p[5]) * f * f + f + one;
        // Return 2^m * exp(r).
        auto ef =
            tvm::reinterpret(DataType::Float(32), ::tvm::cast(DataType::Int(32), n + b) << 23);
        return ::tvm::max(ef * y, _x(i));  // NOLINT(*)
      },
      name, tag);
}

/*!
 * \brief Fast exponential function implementation
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is exponent operation
 *
 */
inline Tensor fast_exp(const Tensor& x, std::string name = "T_fast_exp",
                       std::string tag = kElementWise) {
  if (x->dtype == DataType::Float(32)) {
    auto ret = fast_exp_float32(x, name, tag);
    return ret;
  } else {
    return compute(
        x->shape, [&](const Array<Var>& i) { return ::tvm::exp(x(i)); }, name, tag);
  }
}

/*!
 * \brief Fast_erf_float expression from Eigen
 */
inline Tensor fast_erf_float32(const Tensor& data, std::string name, std::string tag) {
  return compute(
      data->shape, [&](const Array<Var>& i) { return fast_erf_float_expr(data(i), 32); }, name,
      tag);
}

/*!
 * \brief Fast_erf_float expression from Eigen for float16.
 */
inline Tensor fast_erf_float16(const Tensor& data, std::string name, std::string tag) {
  return compute(
      data->shape, [&](const Array<Var>& i) { return fast_erf_float_expr(data(i), 16); }, name,
      tag);
}

/*!
 * \brief Fast erf implementation
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is erf operation
 */
inline Tensor fast_erf(const Tensor& x, std::string name = "T_fast_erf",
                       std::string tag = kElementWise) {
  if (x->dtype == DataType::Float(32)) {
    auto ret = fast_erf_float32(x, name, tag);
    return ret;
  } else if (x->dtype == DataType::Float(16)) {
    auto ret = fast_erf_float16(x, name, tag);
    return ret;
  } else {
    return topi::erf(x);
  }
}

}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_ELEMWISE_H_
