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
 * \file unwrap_vector_expr.h
 *
 * \brief Centralized location for extraction of constraints from a boolean expression.
 */

#ifndef TVM_ARITH_UNWRAP_VECTOR_EXPR_H_
#define TVM_ARITH_UNWRAP_VECTOR_EXPR_H_

#include <tvm/tir/expr.h>

#include <vector>

namespace tvm {
namespace arith {

/* \brief Unwraps a component of a vector expression
 *
 * Utility to break up a vector expression into a specific component
 * of the expression.
 *
 * Example: `Ramp(start, stride, n)` => `start + stride*lane`
 * Example: `Broadcast(value, n)` => `value`
 * Example: `2*Ramp(start, stride, n) + Broadcast(value,n)` => `2*(start + stride*lane) + value`
 *
 * \param vector_expr The vectorized expression to examine
 *
 * \param lane Which lane of the vectorized expression to extract.
 *
 * \returns A scalar expression
 */
PrimExpr UnwrapVectorExpr(const PrimExpr& vector_expr, const PrimExpr& lane);

}  // namespace arith
}  // namespace tvm

#endif  // TVM_ARITH_UNWRAP_VECTOR_EXPR_H_
