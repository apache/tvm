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

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

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
  explicit IRMutatorWithAnalyzer(Analyzer* analyzer) : analyzer_(analyzer) {}

  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  // override functions that need to populate the context information.
  tir::Stmt VisitStmt_(const tir::ForNode* op) override;
  tir::Stmt VisitStmt_(const tir::BlockNode* op) override;
  tir::Stmt VisitStmt_(const tir::LetStmtNode* op) override;
  tir::Stmt VisitStmt_(const tir::IfThenElseNode* op) override;
  tir::Stmt VisitStmt_(const tir::AttrStmtNode* op) override;
  tir::Stmt VisitStmt_(const tir::AssertStmtNode* op) override;
  PrimExpr VisitExpr_(const tir::LetNode* op) override;
  PrimExpr VisitExpr_(const tir::SelectNode* op) override;
  PrimExpr VisitExpr_(const tir::CallNode* op) override;
  PrimExpr VisitExpr_(const tir::ReduceNode* op) override;

 protected:
  /*!
   * \brief Mark the all the buffer shape values in the buffer map as positive value.
   *
   * \note call this function before Visit function's body to maximize
   *       simplification efficiency
   */
  void MarkBufferMapShapes(const tir::PrimFunc& func);

  /*!
   * \brief Use internal bound information to perform inter map simplification of indices.
   * \note Only do this during layout remapping
   */
  Array<PrimExpr> IterMapSimplifyWithContext(const Array<PrimExpr>& indices, bool non_trivial_only);

  /*! \brief internal analyzer field. */
  Analyzer* analyzer_;
  // the following two fields are useful in case we want
  // note however that iter map analysis are usually more
  // expensive and we only encourage doing them during
  // necessary cases like layout remapping
  /*! \brief Recorded loop iterators */
  Map<Var, Range> iter_vars_;
  /*! \brief iterator predicates */
  Array<PrimExpr> iter_predicates_;
  /*!
   * \brief Run callback while trying to record iter predicate
   * \param conditon Condition to be checked.
   * \param callback The callback to be called.
   */
  template <typename FLambda>
  void WithRecordIterPredicate(PrimExpr condition, FLambda callback) {
    auto f_use_itervar = [this](const tir::VarNode* v) {
      return iter_vars_.count(GetRef<tir::Var>(v));
    };
    // simple heuristics for detecting predicate
    if (tir::UsesVar(condition, f_use_itervar)) {
      iter_predicates_.push_back(condition);
      callback();
      iter_predicates_.pop_back();
    } else {
      callback();
    }
  }
};
}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_IR_MUTATOR_WITH_ANALYZER_H_
