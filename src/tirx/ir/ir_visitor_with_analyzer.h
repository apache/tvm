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
 * \file tirx/ir/ir_visitor_with_analyzer.h
 * \brief IR visitor class with an analyzer context.
 */

#ifndef TVM_TIRX_IR_IR_VISITOR_WITH_ANALYZER_H_
#define TVM_TIRX_IR_IR_VISITOR_WITH_ANALYZER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/ir/with_context.h>
#include <tvm/tirx/stmt_functor.h>

#include <vector>

namespace tvm {
namespace tirx {

class IRVisitorWithAnalyzer : public StmtExprVisitor {
 public:
  using StmtExprVisitor::VTable;
  // Dialect-owned handlers use nested access to the active traversal context.
  class Extension;
  // Extensions register during library initialization, before constructing a visitor.
  static void RegisterExtension(void (*init)(VTable*));
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(IRVisitorWithAnalyzer, StmtExprVisitor)

  PrimExpr Simplify(const PrimExpr& expr) { return analyzer_->Simplify(expr); }

  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const BindNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const IfThenElseNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const AttrStmtNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const AssertStmtNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* op);

  // IRVisitorWithAnalyzer deliberately does not handle Select nodes,
  // because both sides of a Select node are visited regardless of the
  // condition.

 protected:
  static void InitVTable(VTable* vtable);
  explicit IRVisitorWithAnalyzer(const VTable* vtable) : StmtExprVisitor(vtable) {}
  /*! \brief internal analyzer field. */
  arith::Analyzer analyzer_;

  /*! \brief Scope stack for accumulated assert constraints. */
  ScopeStack<WithGroup<arith::ConstraintContext>> constraint_scope_;

  /*! \brief Extract a constraint from a conditional statement
   *
   * Intended for preparing argument for use in
   * `With<ConstraintContext>`.
   */
  PrimExpr ExtractRealCondition(PrimExpr condition) const;
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_IR_IR_VISITOR_WITH_ANALYZER_H_
