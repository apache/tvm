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
 * \file simplify_base.h
 * \brief Expression mutator base class for sym simplifiers.
 */
#ifndef TVM_SYM_SIMPLIFY_BASE_H_
#define TVM_SYM_SIMPLIFY_BASE_H_

#include <tvm/ir/expr_functor.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/ir/with_context.h>
#include <tvm/sym/analyzer.h>

#include <utility>

namespace tvm {
namespace sym {

/*!
 * \brief Arithmetic semantic-operand traversal and analyzer constraint handling.
 *
 * The native arithmetic expression hooks leave types and metadata unchanged.
 * This policy and branch-constraint handling are kept in a dedicated base so
 * ordinary expression mutation retains full traversal. Extension nodes retain
 * the inherited structural fallback.
 */
class SimplifierBase : public tvm::ExprMutator {
 public:
  using Parent = tvm::ExprMutator;
  explicit SimplifierBase(AnalyzerObj* analyzer) : analyzer_(analyzer) {}

  using Parent::Mutate_;

  UnchangedOr<Expr> Mutate_(const OpaqueExprNode* op, InplaceMode inplace_mode) override {
    return ffi::Unchanged();
  }

  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) override {
    return ffi::Unchanged();
  }
  UnchangedOr<Expr> Mutate_(const TupleNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const TupleGetItemNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;

 protected:
  AnalyzerObj* analyzer_;
  ScopeStack<WithGroup<ConstraintContext>> constraint_scope_;
};

}  // namespace sym
}  // namespace tvm
#endif  // TVM_SYM_SIMPLIFY_BASE_H_
