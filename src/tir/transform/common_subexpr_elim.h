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

#ifndef TVM_TIR_TRANSFORM_COMMON_SUBEXPR_ELIM_H_
#define TVM_TIR_TRANSFORM_COMMON_SUBEXPR_ELIM_H_

#include <tvm/ir/scope_stack.h>
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

  Stmt VisitStmt_(const BindNode* op) override;
  Stmt VisitStmt_(const SeqStmtNode* op) override;
  Stmt VisitStmt_(const ForNode* op) override;
  Stmt VisitStmt_(const IfThenElseNode* op) override;
  Stmt VisitStmt_(const AttrStmtNode* op) override;
  Stmt VisitStmt_(const AllocBufferNode* op) override;
  Stmt VisitStmt_(const DeclBufferNode* op) override;
  Stmt VisitStmt_(const WhileNode* op) override;

 private:
  /*! \brief Scope level for the context stack.
   *
   * Each scope level records the size of `context_` when the scope was entered.
   * When the scope exits (via ScopeStack::WithNewScope), the destructor truncates
   * `context_` back to the saved size, automatically cleaning up any context
   * entries added within that scope (e.g., from BindNode or loop variables).
   *
   * This approach keeps `context_` as a flat vector for efficient searching
   * while using ScopeStack for automatic scope-based cleanup.
   */
  struct ContextScopeLevel {
    Context* context{nullptr};
    size_t saved_size{0};

    ContextScopeLevel() = default;
    ContextScopeLevel(const ContextScopeLevel&) = delete;
    ContextScopeLevel& operator=(const ContextScopeLevel&) = delete;
    ContextScopeLevel(ContextScopeLevel&& other) noexcept
        : context(other.context), saved_size(other.saved_size) {
      other.context = nullptr;  // prevent other's destructor from truncating
    }
    ContextScopeLevel& operator=(ContextScopeLevel&& other) noexcept {
      if (this != &other) {
        // Run our cleanup first
        if (context) context->resize(saved_size);
        context = other.context;
        saved_size = other.saved_size;
        other.context = nullptr;
      }
      return *this;
    }

    ~ContextScopeLevel() {
      if (context) context->resize(saved_size);
    }
  };

  /*! \brief Enter a new context scope, recording the current context size.
   *
   * Must be called inside context_scope_.WithNewScope() to initialize the
   * newly-pushed scope level. On scope exit, the destructor of
   * ContextScopeLevel will truncate context_ back to this size.
   */
  void EnterContextScope() {
    auto& level = context_scope_.Current();
    level.context = &context_;
    level.saved_size = context_.size();
  }

  Stmt initial_body_;  // Kept for checking if names of new variables already exist

  /*! \brief Flat context vector associating variables to (optional) definitions.
   *
   * This is the searchable context: VisitExpr and VisitStmt scan it linearly
   * to find existing variables whose values match a candidate computation.
   * Entries are added by BindNode (with a value) and ForNode (loop var, no value).
   * Cleanup is automatic via context_scope_: when a scope exits, context_ is
   * truncated to the size it had when the scope was entered.
   */
  Context context_;

  /*! \brief Scope stack for automatic context cleanup.
   *
   * Body-carrying statements (For, IfThenElse, AttrStmt, Allocate, DeclBuffer,
   * While) create new scope levels via WithNewScope. BindNode entries persist
   * across SeqStmt siblings within the same scope and are cleaned up when the
   * enclosing body-carrying statement's scope exits.
   *
   * The initial scope level (created by ScopeStack's constructor) holds the
   * function parameters added during PerformCSE.
   */
  ScopeStack<ContextScopeLevel> context_scope_;

  int num_last_try_ = 0;  // Number of the last variable tried
  int nb_var_ = 0;        // Number of variables introduced by the CSE pass

  bool identify_equiv_terms_ = false;

  static bool ForbiddenComputation(const PrimExpr& expr);
  static bool IsEligibleComputation(const PrimExpr& expr);
  static bool CanContainEligibleComputations(const PrimExpr& expr);
  static bool OrderOnExprAndFrequency(const std::pair<PrimExpr, size_t>& a,
                                      const std::pair<PrimExpr, size_t>& b);
  Var GenerateNewVar(DataType type_annotation);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORM_COMMON_SUBEXPR_ELIM_H_
