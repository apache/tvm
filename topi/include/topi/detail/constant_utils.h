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
* \file constant_utils.h
* \brief Utility functions for handling constants in TVM expressions
*/
#ifndef TOPI_DETAIL_CONSTANT_UTILS_H_
#define TOPI_DETAIL_CONSTANT_UTILS_H_

#include <string>
#include <vector>

#include "tvm/expr.h"
#include "tvm/ir_pass.h"

namespace topi {
namespace detail {
using namespace tvm;

/*!
 * \brief Test whether the given Expr is a constant integer
 *
 * \param expr the Expr to query
 *
 * \return true if the given expr is a constant int or uint, false otherwise.
 */
inline bool IsConstInt(Expr expr) {
  return
    expr->derived_from<tvm::ir::IntImm>() ||
    expr->derived_from<tvm::ir::UIntImm>();
}

/*!
 * \brief Get the value of the given constant integer expression. An error
 * is logged if the given expression is not a constant integer.
 *
 * \param expr The expression to get the value of
 *
 * \return The integer value.
 */
inline int64_t GetConstInt(Expr expr) {
  if (expr->derived_from<tvm::ir::IntImm>()) {
    return expr.as<tvm::ir::IntImm>()->value;
  }
  if (expr->derived_from<tvm::ir::UIntImm>()) {
    return expr.as<tvm::ir::UIntImm>()->value;
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
inline std::vector<int> GetConstIntValues(Array<Expr> exprs, const std::string& var_name) {
  std::vector<int> result;
  if (!exprs.defined()) return result;
  for (auto expr : exprs) {
    CHECK(IsConstInt(expr)) << "All elements of " << var_name << " must be constant integers";
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
inline std::vector<int64_t> GetConstInt64Values(Array<Expr> exprs, const std::string& var_name) {
  std::vector<int64_t> result;
  if (!exprs.defined()) return result;
  for (auto expr : exprs) {
    CHECK(IsConstInt(expr)) << "All elements of " << var_name << " must be constant integers";
    result.push_back(GetConstInt(expr));
  }
  return result;
}

/*!
 * \brief Check weather the two expressions are equal or not, if not simplify the expressions and check again
 * \note This is stronger equality check than tvm::ir::Equal
 *
 * \param lhs First expreesion
 * \param rhs Second expreesion
 *
 * \return result True if both expressions are equal, else false
 */
inline bool EqualCheck(Expr lhs, Expr rhs) {
  bool result = tvm::ir::Equal(lhs, rhs);
  if (!result) {
    Expr zero(0);
    result = tvm::ir::Equal(tvm::ir::CanonicalSimplify(lhs-rhs), zero);
  }
  return result;
}

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_CONSTANT_UTILS_H_
