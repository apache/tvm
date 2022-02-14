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

#include <tvm/tir/expr.h>

#include <vector>

namespace tvm {
namespace tir {

/*!
 * \brief Toplevel (static) function that tells if an expression contains a subexpression that
          satisfies a given predicate.
 * \param expr The expression to check
 * \param predicate The predicate that must be satisfied
 * \return Whether `expr` contains a subexpression that satisfies `predicate`
 */
bool CheckContains::ExprContains(const PrimExpr& expr,
                                 std::function<bool(const PrimExpr&)> predicate) {
  CheckContains check_contains(predicate);
  check_contains.VisitExpr(expr);
  return check_contains.contains_it_;
}

/*!
 * \brief Toplevel (static) function that tells if a statement contains a subexpression that
          satisfies a given predicate.
 * \param stmt The statement to check
 * \param predicate The predicate that must be satisfied
 * \return Whether `stmt` contains a subexpression that satisfies `predicate`
 */
bool CheckContains::StmtContains(const Stmt& stmt, std::function<bool(const PrimExpr&)> predicate) {
  CheckContains check_contains(predicate);
  check_contains.VisitStmt(stmt);
  return check_contains.contains_it_;
}

/*!
 * \brief Protected constructor of CheckContains.
 * \param predicate The predicate that must be satisfied
 */
CheckContains::CheckContains(std::function<bool(const PrimExpr&)> predicate)
    : predicate_(predicate) {}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprVisitor for expressions.
 * \param expr The expression to visit
 */
void CheckContains::VisitExpr(const PrimExpr& expr) {
  // If the predicate holds on `expr`, we know `expr` contains something which makes
  // the predicate hold
  if (predicate_(expr)) {
    contains_it_ = true;
  } else {
    // Otherwise we continue to look for it recursively by calling the dispatcher
    StmtExprVisitor::VisitExpr(expr);
  }
}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprVisitor for statements.
 * \param stmt The statement to visit
 */
void CheckContains::VisitStmt(const Stmt& stmt) {
  // We keep exploring only if `contains_it_` is false
  if (!contains_it_) {
    // and in order to do that we call the general dispatcher
    StmtExprVisitor::VisitStmt(stmt);
  }
  // As otherwise we already have our answer
}

}  // namespace tir
}  // namespace tvm
