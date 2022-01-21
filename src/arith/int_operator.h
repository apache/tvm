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
 * \file int_operator.h
 * \brief Additional useful operators for integer.
 */
#ifndef TVM_ARITH_INT_OPERATOR_H_
#define TVM_ARITH_INT_OPERATOR_H_

#include <limits>
#include <utility>

namespace tvm {
namespace arith {

/*!
 * \brief Check if an integer op with operand x, y will overflow.
 * \param x The left operand.
 * \param y The left operand.
 * \param min_value The minimum value of the domain.
 * \param max_value The maximum value of the domain.
 * \return Whether overflow can happen.
 * \tparam Op The integer operator.
 */
template <typename Op>
inline bool WillOverflow(int64_t x, int64_t y, int64_t min_value, int64_t max_value) {
  return false;
}

template <>
inline bool WillOverflow<tir::AddNode>(int64_t x, int64_t y, int64_t min_value, int64_t max_value) {
  if ((y > 0) && (x > max_value - y)) return true;
  if ((y < 0) && (x < min_value - y)) return true;
  return false;
}

template <>
inline bool WillOverflow<tir::SubNode>(int64_t x, int64_t y, int64_t min_value, int64_t max_value) {
  if ((y > 0) && (x < min_value + y)) return true;
  if ((y < 0) && (x > max_value + y)) return true;
  return false;
}

template <>
inline bool WillOverflow<tir::MulNode>(int64_t x, int64_t y, int64_t min_value, int64_t max_value) {
  if (y == 0) return false;
  if (y > 0) {
    if (x < min_value / y) return true;
    if (x > max_value / y) return true;
  } else {
    if (y == -1 && x == std::numeric_limits<int64_t>::min()) return true;
    if (x > min_value / y) return true;
    if (x < max_value / y) return true;
  }
  return false;
}

template <>
inline bool WillOverflow<tir::ModNode>(int64_t x, int64_t y, int64_t min_value, int64_t max_value) {
  return y == 0;
}

/*!
 * \brief Perform trunc division of two integers.
 * \param x The left operand.
 * \param y The right operand.
 * \return the result.
 */
inline int64_t truncdiv(int64_t x, int64_t y) { return x / y; }

/*!
 * \brief Compute the truncdiv remainder of two integers.
 * \param x The left operand.
 * \param y The right operand.
 * \return the result.
 */
inline int64_t truncmod(int64_t x, int64_t y) { return x % y; }

/*!
 * \brief Perform floor division of two integers.
 * \param x The left operand.
 * \param y The right operand.
 * \return the result.
 */
inline int64_t floordiv(int64_t x, int64_t y) {
  int64_t rdiv = x / y;
  int64_t rmod = x % y;
  bool is_floor_div = (y >= 0 && rmod >= 0) || (y < 0 && rmod <= 0);
  return is_floor_div ? rdiv : (rdiv - 1);
}

/*!
 * \brief Compute the floordiv remainder of two integers.
 * \param x The left operand.
 * \param y The right operand.
 * \return the result.
 */
inline int64_t floormod(int64_t x, int64_t y) {
  int64_t rmod = x % y;
  bool is_floor_div = (y >= 0 && rmod >= 0) || (y < 0 && rmod <= 0);
  return is_floor_div ? rmod : rmod + y;
}

/*!
 * \brief Use Extended Euclidean algorithm to solve ax + by = gcd(a, b)
 * \param a The first coefficient.
 * \param b The second coefficient.
 * \param x The solution of x.
 * \param y The solution of y.
 * \return The GCD of a and b.
 */
inline int64_t ExtendedEuclidean(int64_t a, int64_t b, int64_t* x, int64_t* y) {
  // Extended Euclidean algorithm
  // if a < 0, the problem can be convert into
  // |a|* (-x) + b * y = gcd(|a|, b)
  //
  // initial condition:
  // a * 0 + b * 1 = b
  // a * 1 + b * 0 = a
  int64_t s = 0, old_s = 1;
  int64_t r = b, old_r = a >= 0 ? a : -a;
  // Iteration (r2 < r1):
  // a * x1 + b * y1 = r1
  // a * x2 + b * y2 = r2
  // The above two eqs can derive the following eq (q = r1 / r2)
  // a * (x1 - x2 * q) + b * (y1 - y2 * q) = r1 - r2 * q = r3
  // Because r3 < r2, the iteration can eventually terminate
  while (r != 0) {
    int64_t q = old_r / r;
    int64_t tmp = old_r;
    old_r = r;
    r = tmp - q * r;
    tmp = old_s;
    old_s = s;
    s = tmp - q * s;
  }

  *x = a >= 0 ? old_s : -old_s;
  if (b != 0) {
    *y = (old_r - (*x) * a) / b;
  } else {
    *y = 1;
  }

  return old_r;
}

/*!
 * \brief Take GCD of a and b.
 * \param a The first operand.
 * \param b The second operand.
 * \return The result.
 */
inline int64_t ZeroAwareGCD(int64_t a, int64_t b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  if (a < b) std::swap(a, b);
  if (b == 0) return a;
  // perform GCD (greatest common divisor)
  // ax + by = gcd(a, b) z if a != 0, b != 0
  while (a % b != 0) {
    a = a % b;
    std::swap(a, b);
  }
  return b;
}

/*!
 * \brief Calculate the least common multiple for two values.
 * \param a an integer number
 * \param b an integer number
 * \return the least common multiple.
 */
inline int64_t LeastCommonMultiple(int64_t a, int64_t b) {
  int64_t x, y;
  return (a * b) / ExtendedEuclidean(a, b, &x, &y);
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_INT_OPERATOR_H_
