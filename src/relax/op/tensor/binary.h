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
 * \file binary.h
 * \brief The functions to make Relax binary arithmetic and comparison operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_BINARY_H_
#define TVM_RELAX_OP_TENSOR_BINARY_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/***************** Arithmetic operators *****************/

/*! \brief Addition with numpy-style broadcasting. */
Expr add(Expr x1, Expr x2);

/*! \brief Division with numpy-style broadcasting. */
Expr divide(Expr x1, Expr x2);

/*! \brief Floor division with numpy-style broadcasting. */
Expr floor_divide(Expr x1, Expr x2);

/*! \brief Log Add Exponent with numpy-style broadcasting. */
Expr log_add_exp(Expr x1, Expr x2);

/*! \brief Multiplication with numpy-style broadcasting. */
Expr multiply(Expr x1, Expr x2);

/*! \brief Power with numpy-style broadcasting. */
Expr power(Expr x1, Expr x2);

/*! \brief Atan2 with numpy-style broadcasting. */
Expr atan2(Expr x1, Expr x2);

/*! \brief Subtraction with numpy-style broadcasting. */
Expr subtract(Expr x1, Expr x2);

/*! \brief Modulo with numpy-style broadcasting. */
Expr mod(Expr x1, Expr x2);

/*! \brief Floor modulo with numpy-style broadcasting. */
Expr floor_mod(Expr x1, Expr x2);

/***************** Comparison operators *****************/

/*! \brief Broadcasted element-wise test for (lhs == rhs). */
Expr equal(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise test for (lhs > rhs). */
Expr greater(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise test for (lhs >= rhs). */
Expr greter_equal(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise test for (lhs < rhs). */
Expr less(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise test for (lhs <= rhs). */
Expr less_equal(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise test for (lhs != rhs). */
Expr not_equal(Expr x1, Expr x2);

/***************** Min/Max *****************/

/*! \brief Element-wise minimum */
Expr minimum(Expr x1, Expr x2);

/*! \brief Element-wise maximum */
Expr maximum(Expr x1, Expr x2);

/***************** Logical operators *****************/

/*! \brief Broadcasted element-wise logical and */
Expr logical_and(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise logical or */
Expr logical_or(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise logical xor */
Expr logical_xor(Expr x1, Expr x2);

/***************** Bitwise operators *****************/

/*! \brief Broadcasted element-wise bitwise and */
Expr bitwise_and(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise bitwise or */
Expr bitwise_or(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise bitwise xor */
Expr bitwise_xor(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise bitwise shift left */
Expr left_shift(Expr x1, Expr x2);

/*! \brief Broadcasted element-wise bitwise shift right */
Expr right_shift(Expr x1, Expr x2);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_BINARY_H_
