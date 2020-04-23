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
 * \file tvm/arithmetic/ir_mutator_with_analyzer.h
 * \brief IR mutator base-class with an analyzer context.
 */
#ifndef TVM_ARITH_IR_MUTATOR_WITH_ANALYZER_H_
#define TVM_ARITH_IR_MUTATOR_WITH_ANALYZER_H_

#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include <utility>

namespace tvm {
namespace arith {

/*!
 * \brief IRMutator with an analyzer context.
 *
 * This class can sub-classed by ir mutators that need an analyzer.
 * It will populates scope-related info such as bounds of loop-variables and constraints
 * for the analyzer, so that the child class can do accurate context-dependent analysis.
 *
 * \sa src/arithmetic/ir_mutator_with_analyzer.cc
 */
class IRMutatorWithAnalyzer : public tir::StmtExprMutator {
 public:
  explicit IRMutatorWithAnalyzer(Analyzer* analyzer)
      : analyzer_(analyzer) {}

  using StmtExprMutator::VisitStmt_;
  using StmtExprMutator::VisitExpr_;

  // override functions that need to populate the context information.
  tir::Stmt VisitStmt_(const tir::ForNode* op) override;
  tir::Stmt VisitStmt_(const tir::LetStmtNode* op) override;
  tir::Stmt VisitStmt_(const tir::IfThenElseNode* op) override;
  tir::Stmt VisitStmt_(const tir::AttrStmtNode* op) override;
  tir::Stmt VisitStmt_(const tir::AssertStmtNode* op) override;
  PrimExpr VisitExpr_(const tir::LetNode* op) override;
  PrimExpr VisitExpr_(const tir::SelectNode* op) override;
  PrimExpr VisitExpr_(const tir::CallNode* op) override;
  PrimExpr VisitExpr_(const tir::ReduceNode* op) override;

 protected:
  /*! \brief internal analyzer field. */
  Analyzer* analyzer_;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_IR_MUTATOR_WITH_ANALYZER_H_
