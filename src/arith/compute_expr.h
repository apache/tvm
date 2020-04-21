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
 * \file compute_expr.h
 * \brief Utility to invoke certan compute operations.
 */
#ifndef TVM_ARITH_COMPUTE_EXPR_H_
#define TVM_ARITH_COMPUTE_EXPR_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <limits>
#include <algorithm>

namespace tvm {
namespace arith {

/*!
 * \brief Compute the expression with the given binary op.
 * \param lhs The left operand
 * \param rhs The right operand
 * \tparam Op the computation operator
 * \return The result.
 */
template<typename OP>
inline PrimExpr Compute(PrimExpr lhs, PrimExpr rhs) {
  return OP::make(lhs, rhs);
}

/*!
 * \brief Compute an reduction with Op
 * \param values The input values.
 * \param empty_value The value when return if it is empty, can be Expr()
 *        which will cause an error to be rasied.
 * \tparam Op The computation operator
 * \return The result.
 */
template<typename Op>
inline PrimExpr ComputeReduce(
    const Array<PrimExpr>& values, PrimExpr empty_value);

template<>
inline PrimExpr Compute<tir::AddNode>(PrimExpr a, PrimExpr b) {
  return a + b;
}

template<>
inline PrimExpr Compute<tir::SubNode>(PrimExpr a, PrimExpr b) {
  return a - b;
}

template<>
inline PrimExpr Compute<tir::MulNode>(PrimExpr a, PrimExpr b) {
  return a * b;
}

template<>
inline PrimExpr Compute<tir::DivNode>(PrimExpr a, PrimExpr b) {
  return truncdiv(a, b);
}

template<>
inline PrimExpr Compute<tir::ModNode>(PrimExpr a, PrimExpr b) {
  return truncmod(a, b);
}

template<>
inline PrimExpr Compute<tir::MaxNode>(PrimExpr a, PrimExpr b) {
  return max(a, b);
}

template<>
inline PrimExpr Compute<tir::MinNode>(PrimExpr a, PrimExpr b) {
  return min(a, b);
}

template<typename Op>
inline PrimExpr ComputeReduce(const Array<PrimExpr>& values, PrimExpr empty_value) {
  if (values.size() == 0U) {
    CHECK(empty_value.defined());
    return empty_value;
  }
  PrimExpr res = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    res = Compute<Op>(res, values[i]);
  }
  return res;
}

}  // namespace arith
}  // namespace tvm
#endif   // TVM_ARITH_COMPUTE_EXPR_H_
