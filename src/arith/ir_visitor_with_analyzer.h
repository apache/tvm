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
namespace tir {

class IRVisitorWithAnalyzer : public StmtExprVisitor {
 public:
  PrimExpr Simplify(const PrimExpr& expr) { return analyzer_.Simplify(expr); }

  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;

  void VisitStmt_(const ForNode* op);
  void VisitStmt_(const BlockNode* op);
  void VisitStmt_(const LetStmtNode* op);
  void VisitStmt_(const IfThenElseNode* op);
  void VisitStmt_(const AttrStmtNode* op);
  void VisitStmt_(const AssertStmtNode* op);
  void VisitExpr_(const CallNode* op);
  void VisitExpr_(const LetNode* op);
  void VisitExpr_(const SelectNode* op);
  void VisitExpr_(const ReduceNode* op);

 protected:
  /*! \brief internal analyzer field. */
  arith::Analyzer analyzer_;

 private:
  PrimExpr ExtractRealCondition(PrimExpr condition) const;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_ARITH_IR_VISITOR_WITH_ANALYZER_H_
