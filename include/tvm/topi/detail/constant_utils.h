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
 * \file constant_utils.h
 * \brief Utility functions for handling constants in TVM expressions
 */
#ifndef TVM_TOPI_DETAIL_CONSTANT_UTILS_H_
#define TVM_TOPI_DETAIL_CONSTANT_UTILS_H_

#include <tvm/arith/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>

#include <string>
#include <vector>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Test whether the given Expr is a constant integer
 *
 * \param expr the Expr to query
 *
 * \return true if the given expr is a constant int or uint, false otherwise.
 */
inline bool IsConstInt(PrimExpr expr) { return expr->IsInstance<tvm::tir::IntImmNode>(); }

/*!
 * \brief Test whether the given Array has every element as constant integer.
 * Undefined elements are also treat as constants.
 *
 * \param array the array to query
 *
 * \return true if every element in array is constant int or uint, false otherwise.
 */
inline bool IsConstIntArray(Array<PrimExpr> array) {
  bool is_const_int = true;
  for (auto const& elem : array) {
    is_const_int &= !elem.defined() || elem->IsInstance<tvm::tir::IntImmNode>();
  }
  return is_const_int;
}

/*!
 * \brief Get the value of the given constant integer expression. An error
 * is logged if the given expression is not a constant integer.
 *
 * \param expr The expression to get the value of
 *
 * \return The integer value.
 */
inline int64_t GetConstInt(PrimExpr expr) {
  if (expr->IsInstance<tvm::IntImmNode>()) {
    return expr.as<tvm::IntImmNode>()->value;
  }
  LOG(ERROR) << "expr must be a constant integer";
  return -1;
}

/*!
 * \brief Get the value of all the constant integer expressions in the given array
 *
 * \param exprs The array of expressions to get the values of
 * \param var_name The name to be used when logging an error in the event that any
 * of the expressions are not constant integers.
 *
 * \return A vector of the integer values
 */
inline std::vector<int> GetConstIntValues(Array<PrimExpr> exprs, const std::string& var_name) {
  std::vector<int> result;
  if (!exprs.defined()) return result;
  for (auto expr : exprs) {
    ICHECK(IsConstInt(expr)) << "All elements of " << var_name << " must be constant integers";
    result.push_back(GetConstInt(expr));
  }
  return result;
}

/*!
 * \brief Get the value of all the constant integer expressions in the given array
 *
 * \param exprs The array of expressions to get the values of
 * \param var_name The name to be used when logging an error in the event that any
 * of the expressions are not constant integers.
 *
 * \return A vector of the int64_t values
 */
inline std::vector<int64_t> GetConstInt64Values(Array<PrimExpr> exprs,
                                                const std::string& var_name) {
  std::vector<int64_t> result;
  if (!exprs.defined()) return result;
  for (auto expr : exprs) {
    ICHECK(IsConstInt(expr)) << "All elements of " << var_name << " must be constant integers";
    result.push_back(GetConstInt(expr));
  }
  return result;
}

/*!
 * \brief Check whether the two expressions are equal or not, if not simplify the expressions and
 * check again
 * \note This is stronger equality check than tvm::tir::Equal
 * \param lhs First expression
 * \param rhs Second expression
 * \return result True if both expressions are equal, else false
 */
inline bool EqualCheck(PrimExpr lhs, PrimExpr rhs) {
  tvm::tir::ExprDeepEqual expr_equal;
  bool result = expr_equal(lhs, rhs);
  if (!result) {
    PrimExpr t = tvm::arith::Analyzer().Simplify(lhs - rhs);
    if (const IntImmNode* i = t.as<IntImmNode>()) {
      result = i->value == 0;
    }
  }
  return result;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_CONSTANT_UTILS_H_
