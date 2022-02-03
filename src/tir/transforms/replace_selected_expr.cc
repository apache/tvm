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
* \file replace_expr_selected.cc
* \brief Implementation of the pass that replaces in a statement
          or expression all the subexpressions that are selected
          with a predicate by another expression.
*/

#include "replace_selected_expr.h"

#include <tvm/ir/transform.h>  // For the class Pass and the class PassContext
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>  // For the class PrimFunc
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>  // For the declaration of the pass

namespace tvm {
namespace tir {

/*!
 * \brief Toplevel (static) function that replace in an expression
          everything that is selected by a predicate.
 * \param expr The PrimExpr in which replacements will be performed
 * \param new_expr The new expression replacing everything that's selected by the predicate
 * \param predicate_selector The predicate which tells what to replace in `expr`
 * \param can_replace_inside The predicate which tells in which nodes we are allowed to recurse
                              for pursuing further replacements.
 * \return A new expression where the replacements have been done
 */
PrimExpr ReplaceSelectedExpr::ReplaceSelectedExprInExpr(
    const PrimExpr& expr, std::function<bool(const PrimExpr&)> predicate_selector,
    const PrimExpr& new_expr, std::function<bool(const PrimExpr&)> can_replace_inside) {
  ReplaceSelectedExpr replace_expr_selected(predicate_selector, new_expr, can_replace_inside);
  return replace_expr_selected.VisitExpr(expr);
}

/*!
 * \brief Toplevel (static) function that replace in a statement what is selected by a predicate.
 * \param stmt The Stmt in which replacements will be performed
 * \param new_expr The new expression that will replace everything that's selected by the predicate
 * \param predicate_selector The predicate which tells what to replace in `stmt`
 * \param can_replace_inside The predicate which tells in which nodes we are allowed to recurse
                              for pursuing further replacements
 * \return A new statement where the replacements have been done
 */
Stmt ReplaceSelectedExpr::ReplaceSelectedExprInStmt(
    const Stmt& stmt, std::function<bool(const PrimExpr&)> predicate_selector,
    const PrimExpr& new_expr, std::function<bool(const PrimExpr&)> can_replace_inside) {
  ReplaceSelectedExpr replace_expr_selected(predicate_selector, new_expr, can_replace_inside);
  return replace_expr_selected.VisitStmt(stmt);
}

/*!
 * \brief Protected constructor of ReplaceSelectedExpr.
 * \param predicate_selector The predicate which tells what to replace
 * \param new_expr The new expression that will replace everything that's selected by the predicate
 * \param can_replace_inside The predicate which tells in which nodes we are allowed to recurse
                              for pursuing further replacements
 */
ReplaceSelectedExpr::ReplaceSelectedExpr(std::function<bool(const PrimExpr&)> predicate_selector,
                                         const PrimExpr& new_expr,
                                         std::function<bool(const PrimExpr&)> can_replace_inside)
    : predicate_selector_(predicate_selector),
      new_expr_(new_expr),
      can_replace_inside_(can_replace_inside) {}

/*!
 * \brief The method which overrides the generic dispatcher of StmtExprMutator
 * \param expr The expression to mutate
 */
PrimExpr ReplaceSelectedExpr::VisitExpr(const PrimExpr& expr) {
  // If the current expression is selected by the predicate
  if (predicate_selector_(expr)) {
    // Then simply return the new expression
    return new_expr_;
  } else {
    // If replacing inside the current expression is allowed
    if (can_replace_inside_(expr)) {
      // then we continue the exploration recursively
      return StmtExprMutator::VisitExpr(expr);
    } else {
      // otherwise we simply return the current expression
      return expr;
    }
  }
}

}  // namespace tir
}  // namespace tvm
