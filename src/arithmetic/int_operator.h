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
 *  Copyright (c) 2019 by Contributors
 * \file int_operator.h
 * \brief Additional useful operators for integer.
 */
#ifndef TVM_ARITHMETIC_INT_OPERATOR_H_
#define TVM_ARITHMETIC_INT_OPERATOR_H_

#include <limits>

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
template<typename Op>
inline bool WillOverflow(int64_t x,
                         int64_t y,
                         int64_t min_value,
                         int64_t max_value) {
  return false;
}

template<>
inline bool WillOverflow<ir::Add>(int64_t x,
                                  int64_t y,
                                  int64_t min_value,
                                  int64_t max_value) {
  if ((y > 0) && (x > max_value - y)) return true;
  if ((y < 0) && (x < min_value - y)) return true;
  return false;
}

template<>
inline bool WillOverflow<ir::Sub>(int64_t x,
                                  int64_t y,
                                  int64_t min_value,
                                  int64_t max_value) {
  if ((y > 0) && (x < min_value + y)) return true;
  if ((y < 0) && (x > max_value + y)) return true;
  return false;
}

template<>
inline bool WillOverflow<ir::Mul>(int64_t x,
                                  int64_t y,
                                  int64_t min_value,
                                  int64_t max_value) {
  if (y == 0) return false;
  if (y > 0) {
    if (x < min_value / y)  return true;
    if (x > max_value / y)  return true;
  } else {
    if (y == -1 && x == std::numeric_limits<int64_t>::min()) return true;
    if (x > min_value / y)  return true;
    if (x < max_value / y)  return true;
  }
  return false;
}

template<>
inline bool WillOverflow<ir::Mod>(int64_t x,
                                  int64_t y,
                                  int64_t min_value,
                                  int64_t max_value) {
  return y == 0;
}

/*!
 * \brief Peform floor division of two integers.
 * \param x The left operand.
 * \param y The right operand.
 * \return the result.
 */
inline int64_t floordiv(int64_t x, int64_t y) {
  bool round_down =
      (x >= 0 && y >= 0) ||
      (x <= 0 && y <= 0) ||
      (x % y == 0);
  return round_down ? (x / y) : (x / y - 1);
}


/*!
 * \brief Compute the floordiv remainder of two integers.
 * \param x The left operand.
 * \param y The right operand.
 * \return the result.
 */
inline int64_t floormod(int64_t x, int64_t y) {
  bool round_down =
      (x >= 0 && y >= 0) ||
      (x <= 0 && y <= 0) ||
      (x % y == 0);
  return round_down ? (x % y) : (x % y + y);
}

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_INT_OPERATOR_H_
