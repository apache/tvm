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
 * \file tvm/arithmetic/ir_visitor_with_analyzer.h
 * \brief IR visitor class with an analyzer context.
 */

#ifndef TVM_ARITH_IR_VISITOR_WITH_ANALYZER_H_
#define TVM_ARITH_IR_VISITOR_WITH_ANALYZER_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {

class IRVisitorWithAnalyzer : public tir::StmtExprVisitor {
 public:
  PrimExpr Simplify(const PrimExpr& expr) { return analyzer_.Simplify(expr); }

  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;

  void VisitStmt_(const tir::ForNode* op);
  void VisitStmt_(const tir::BlockNode* op);
  void VisitStmt_(const tir::LetStmtNode* op);
  void VisitStmt_(const tir::IfThenElseNode* op);
  void VisitStmt_(const tir::AttrStmtNode* op);
  void VisitStmt_(const tir::AssertStmtNode* op);
  void VisitExpr_(const tir::CallNode* op);
  void VisitExpr_(const tir::LetNode* op);
  void VisitExpr_(const tir::ReduceNode* op);

  // IRVisitorWithAnalyzer deliberately does not handle Select nodes,
  // because both sides of a Select node are visited regardless of the
  // condition.

 protected:
  /*! \brief internal analyzer field. */
  arith::Analyzer analyzer_;

  /*! \brief Extract a constraint from a conditional statement
   *
   * Intended for preparing argument for use in
   * `With<ConstraintContext>`.
   */
  PrimExpr ExtractRealCondition(PrimExpr condition) const;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_IR_VISITOR_WITH_ANALYZER_H_
