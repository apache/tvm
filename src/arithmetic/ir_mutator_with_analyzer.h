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
#ifndef TVM_ARITHMETIC_IR_MUTATOR_WITH_ANALYZER_H_
#define TVM_ARITHMETIC_IR_MUTATOR_WITH_ANALYZER_H_

#include <tvm/ir_functor_ext.h>
#include <tvm/arithmetic.h>
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
class IRMutatorWithAnalyzer : public ir::StmtExprMutator {
 public:
  explicit IRMutatorWithAnalyzer(Analyzer* analyzer)
      : analyzer_(analyzer) {}

  using StmtExprMutator::VisitStmt_;
  using StmtExprMutator::VisitExpr_;

  // override functions that need to populate the context information.
  Stmt VisitStmt_(const ir::For* op) override;
  Stmt VisitStmt_(const ir::LetStmt* op) override;
  Stmt VisitStmt_(const ir::IfThenElse* op) override;
  Stmt VisitStmt_(const ir::AttrStmt* op) override;
  Stmt VisitStmt_(const ir::AssertStmt* op) override;
  Expr VisitExpr_(const ir::Let* op) override;
  Expr VisitExpr_(const ir::Select* op) override;
  Expr VisitExpr_(const ir::Call* op) override;
  Expr VisitExpr_(const ir::Reduce* op) override;

 protected:
  /*! \brief internal analyzer field. */
  Analyzer* analyzer_;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_IR_MUTATOR_WITH_ANALYZER_H_
