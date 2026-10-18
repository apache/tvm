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
 * \brief Integer division and remainder helpers for primitive operations.
 */
#ifndef TVM_IR_PRIM_INT_OPERATOR_H_
#define TVM_IR_PRIM_INT_OPERATOR_H_

#include <cstdint>

namespace tvm {
namespace prim {
namespace detail {

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

}  // namespace detail
}  // namespace prim
}  // namespace tvm
#endif  // TVM_IR_PRIM_INT_OPERATOR_H_
