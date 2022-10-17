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
 * \file replace_selected_expr.h
 * \brief Interface of the pass that replaces in a statement
           or expression all the subexpressions that are selected
           with a predicate by another expression.
 */

#ifndef TVM_TIR_TRANSFORMS_REPLACE_SELECTED_EXPR_H_
#define TVM_TIR_TRANSFORMS_REPLACE_SELECTED_EXPR_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>  // For the class StmtExprMutator

namespace tvm {
namespace tir {

/*!
 * \brief Mutator for replacing the expressions selected by a predicate in a statement and/or
          in an expression, which only replace inside of nodes in which it is allowed to perform
          replacecements (given by a second predicate)
 */
class ReplaceSelectedExpr : public StmtExprMutator {
 public:
  // Toplevel (static) functions
  static PrimExpr ReplaceSelectedExprInExpr(
      const PrimExpr& expr, std::function<bool(const PrimExpr&)> predicate_selector,
      const PrimExpr& new_expr, std::function<bool(const PrimExpr&)> can_replace_inside);
  static Stmt ReplaceSelectedExprInStmt(const Stmt& stmt,
                                        std::function<bool(const PrimExpr&)> predicate_selector,
                                        const PrimExpr& new_expr,
                                        std::function<bool(const PrimExpr&)> can_replace_inside);

 protected:
  // Constructor
  ReplaceSelectedExpr(std::function<bool(const PrimExpr&)> predicate_selector,
                      const PrimExpr& new_expr,
                      std::function<bool(const PrimExpr&)> can_replace_inside);

  PrimExpr VisitExpr(const PrimExpr& expr) override;

 private:
  // The predicate used for selecting what will be replaced
  std::function<bool(const PrimExpr&)> predicate_selector_;
  // The expression used for replacing
  const PrimExpr& new_expr_;
  // The predicate used for knowning inside which nodes we can do rewriting
  // (i.e. in which nodes it can recurse)
  std::function<bool(const PrimExpr&)> can_replace_inside_;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORMS_REPLACE_SELECTED_EXPR_H_
