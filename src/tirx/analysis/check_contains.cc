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
  * \file check_contains.cc
  * \brief Implementation of the analysis that tells if an expression contains
            a node that satisfies a given predicate.
  */

#include "check_contains.h"

#include <tvm/ir/prim/expr.h>

#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief Toplevel (static) function that tells if an expression contains a subexpression that
          satisfies a given predicate.
 * \param expr The expression to check
 * \param predicate The predicate that must be satisfied
 * \return Whether `expr` contains a subexpression that satisfies `predicate`
 */
bool CheckContains::ExprContains(const PrimExpr& expr,
                                 std::function<bool(const PrimExpr&)> predicate) {
  auto check_contains = ffi::make_object<CheckContains>(predicate);
  check_contains->Visit(expr);
  return check_contains->contains_it_;
}

/*!
 * \brief Toplevel (static) function that tells if a statement contains a subexpression that
          satisfies a given predicate.
 * \param stmt The statement to check
 * \param predicate The predicate that must be satisfied
 * \return Whether `stmt` contains a subexpression that satisfies `predicate`
 */
bool CheckContains::StmtContains(const Stmt& stmt, std::function<bool(const PrimExpr&)> predicate) {
  auto check_contains = ffi::make_object<CheckContains>(predicate);
  check_contains->Visit(stmt);
  return check_contains->contains_it_;
}

/*!
 * \brief Protected constructor of CheckContains.
 * \param predicate The predicate that must be satisfied
 */
CheckContains::CheckContains(std::function<bool(const PrimExpr&)> predicate)
    : predicate_(predicate) {}

ffi::Optional<VisitInterrupt> CheckContains::Visit(ffi::AnyView value) {
  if (auto prim_expr = value.as<PrimExpr>(); prim_expr && predicate_(prim_expr.value())) {
    contains_it_ = true;
    return std::nullopt;
  }
  if (value.as<StmtNode>() && contains_it_) return std::nullopt;
  return StmtExprVisitor::Visit(value);
}

}  // namespace tirx
}  // namespace tvm
