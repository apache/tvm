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
#ifndef TVM_ARITHMETIC_COMPUTE_EXPR_H_
#define TVM_ARITHMETIC_COMPUTE_EXPR_H_

#include <tvm/ir.h>
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
inline Expr Compute(Expr lhs, Expr rhs) {
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
inline Expr ComputeReduce(
    const Array<Expr>& values, Expr empty_value);

inline bool GetConst(Expr e, int64_t* out) {
  if (e.type().is_vector()) return false;
  const int64_t* v = as_const_int(e);
  if (v) {
    *out = *v; return true;
  } else {
    return false;
  }
}

// get a small constant int
inline bool GetConstInt(Expr e, int* out) {
  int64_t v1 = 0;
  if (GetConst(e, &v1)) {
    if (v1 > static_cast<int64_t>(
            std::numeric_limits<int>::max())) return false;
    *out = static_cast<int>(v1); return true;
  }
  return false;
}

template<>
inline Expr Compute<ir::Add>(Expr a, Expr b) {
  return a + b;
}

template<>
inline Expr Compute<ir::Sub>(Expr a, Expr b) {
  return a - b;
}

template<>
inline Expr Compute<ir::Mul>(Expr a, Expr b) {
  return a * b;
}

template<>
inline Expr Compute<ir::Div>(Expr a, Expr b) {
  return truncdiv(a, b);
}

template<>
inline Expr Compute<ir::Mod>(Expr a, Expr b) {
  return truncmod(a, b);
}

template<>
inline Expr Compute<ir::Max>(Expr a, Expr b) {
  return max(a, b);
}

template<>
inline Expr Compute<ir::Min>(Expr a, Expr b) {
  return min(a, b);
}

template<typename Op>
inline Expr ComputeReduce(const Array<Expr>& values, Expr empty_value) {
  if (values.size() == 0U) {
    CHECK(empty_value.defined());
    return empty_value;
  }
  Expr res = values[0];
  for (size_t i = 1; i < values.size(); ++i) {
    res = Compute<Op>(res, values[i]);
  }
  return res;
}

}  // namespace arith
}  // namespace tvm
#endif   // TVM_ARITHMETIC_COMPUTE_EXPR_H_
