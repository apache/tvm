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
 * \file common_subexpr_elim.h
 * \brief Interface of the Common Subexpressions Elimination (CSE) pass which rewrites statements
           and expressions in order to eliminate redundant computations. In order to achieve that,
           common (sub-)expressions are introduced into variables with let-in bindings, and the
           places where the expression was used are replaced with the freshly introduced variable.
 */

#ifndef TVM_TIR_TRANSFORMS_COMMON_SUBEXPR_ELIM_H_
#define TVM_TIR_TRANSFORMS_COMMON_SUBEXPR_ELIM_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>  // For the class StmtExprMutator
#include <tvm/tir/var.h>

#include <utility>  // For std::pair
#include <vector>

#include "common_subexpr_elim_tools.h"  // For the class MaybeValue

namespace tvm {
namespace tir {

/*!
 * \brief A context is a vector of pairs that associates Var to MaybeValue
          (which are either an expression or nothing)
 */
using Context = std::vector<std::pair<Var, MaybeValue>>;

/*!
 * \brief Mutator that performs Common Subexpression Elimination (CSE) for the body of a
          PrimFunc, mutating both its expressions and statements.
 */
class CommonSubexpressionEliminator : public StmtExprMutator {
 public:
  // Toplevel (static) function
  static Stmt PerformCSE(const Stmt& stmt, const Context& context_init, bool identify_equiv_terms);

  PrimExpr VisitExpr(const PrimExpr& expr) override;
  Stmt VisitStmt(const Stmt& stmt) override;

  int GetNbVarGenerated();

 protected:
  // Constructor
  CommonSubexpressionEliminator(const Stmt& stmt, const Context& context_init,
                                bool identify_equiv_terms);

  PrimExpr VisitExpr_(const LetNode* op) override;

  Stmt VisitStmt_(const LetStmtNode* op) override;
  Stmt VisitStmt_(const ForNode* op) override;

 private:
  Stmt initial_body_;     // Kept for checking if names of new variables already exist
  Context context_;       // Context associating variables to (maybe) definitions
  int num_last_try_ = 0;  // Number of the last variable tried
  int nb_var_ = 0;        // Number of variables introduced by the CSE pass

  bool identify_equiv_terms_ = false;

  static bool ForbiddenComputation(const PrimExpr& expr);
  static bool IsEligibleComputation(const PrimExpr& expr);
  static bool CanContainEligibleComputations(const PrimExpr& expr);
  static bool OrderOnExprAndFrequency(std::pair<PrimExpr, size_t> a, std::pair<PrimExpr, size_t> b);
  Var GenerateNewVar(DataType type_annotation);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORMS_COMMON_SUBEXPR_ELIM_H_
