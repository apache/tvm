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
 * \file tvm/tirx/stmt_functor_with_analyzer.h
 * \brief Statement and expression functors with an analyzer context.
 */

#ifndef TVM_TIRX_STMT_FUNCTOR_WITH_ANALYZER_H_
#define TVM_TIRX_STMT_FUNCTOR_WITH_ANALYZER_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/ir/with_context.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_set>
#include <utility>

namespace tvm {
namespace tirx {

class TVM_TIRX_DLL StmtExprVisitorWithAnalyzer : public StmtExprVisitor {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitorWithAnalyzer, StmtExprVisitor)

  PrimExpr Simplify(const PrimExpr& expr) { return analyzer_->Simplify(expr); }

  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const BindNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const IfNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const AssertStmtNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op);
  ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* op);

  // StmtExprVisitorWithAnalyzer deliberately does not handle Select nodes,
  // because both sides of a Select node are visited regardless of the
  // condition.

 protected:
  static void InitVTable(VTable* vtable);
  explicit StmtExprVisitorWithAnalyzer(const VTable* vtable) : StmtExprVisitor(vtable) {}
  /*! \brief internal analyzer field. */
  sym::Analyzer analyzer_;

  /*! \brief Scope stack for accumulated assert constraints. */
  ScopeStack<WithGroup<sym::ConstraintContext>> constraint_scope_;

  /*! \brief Extract a constraint from a conditional statement
   *
   * Intended for preparing argument for use in
   * `With<ConstraintContext>`.
   */
  PrimExpr ExtractRealCondition(PrimExpr condition) const;
};

/*!
 * \brief IRMutator with an analyzer context.
 *
 * This class can sub-classed by ir mutators that need an analyzer.
 * It will populates scope-related info such as bounds of loop-variables and constraints
 * for the analyzer, so that the child class can do accurate context-dependent analysis.
 *
 * \sa src/tirx/ir/ir_mutator_with_analyzer.cc
 */
class TVM_TIRX_DLL StmtExprMutatorWithAnalyzer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit StmtExprMutatorWithAnalyzer(const sym::Analyzer& analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer.get()) {}
  explicit StmtExprMutatorWithAnalyzer(sym::AnalyzerObj* analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer, GlobalVTable()) {}

  // override functions that need to populate the context information.
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const AssertStmtNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;

 protected:
  static void InitVTable(VTable* vtable);
  StmtExprMutatorWithAnalyzer(sym::AnalyzerObj* analyzer, const VTable* vtable)
      : StmtExprMutator(vtable), analyzer_(analyzer) {}
  static const VTable* GlobalVTable();
  /*!
   * \brief Mark all tensor-parameter shape values as positive values.
   *
   * \note call this function before Visit function's body to maximize
   *       simplification efficiency
   */
  void MarkTensorParamShapes(const Function& func);

  /*!
   * \brief Use internal bound information to perform inter map simplification of indices.
   * \note Only do this during layout remapping
   */
  ffi::Array<PrimExpr> IterMapSimplifyWithContext(const ffi::Array<PrimExpr>& indices,
                                                  bool non_trivial_only);

  /*! \brief internal analyzer field. */
  sym::AnalyzerObj* analyzer_;
  /*! \brief Scope stack for accumulated assert constraints. */
  ScopeStack<WithGroup<sym::ConstraintContext>> constraint_scope_;
  // the following two fields are useful in case we want
  // note however that iter map analysis are usually more
  // expensive and we only encourage doing them during
  // necessary cases like layout remapping
  /*! \brief Recorded loop iterators */
  ffi::Map<PrimVar, Range> iter_vars_;
  /*! \brief iterator predicates */
  ffi::Array<PrimExpr> iter_predicates_;
  /*!
   * \brief Run callback while trying to record iter predicate
   * \param conditon Condition to be checked.
   * \param callback The callback to be called.
   */
  template <typename FLambda>
  void WithRecordIterPredicate(PrimExpr condition, FLambda callback) {
    std::unordered_set<const VarNode*> iter_var_nodes;
    for (const auto& [var, _] : iter_vars_) {
      iter_var_nodes.insert(var.get());
    }
    auto f_use_itervar = [&iter_var_nodes](const tvm::VarNode* v) {
      return iter_var_nodes.count(v);
    };
    auto walkfn = [&](const tvm::Var& var) -> ffi::Expected<ffi::WalkResult> {
      return f_use_itervar(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                      : ffi::WalkResult::Advance();
    };
    // simple heuristics for detecting predicate
    if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(condition, walkfn).has_value()) {
      iter_predicates_.push_back(condition);
      callback();
      iter_predicates_.pop_back();
    } else {
      callback();
    }
  }
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_STMT_FUNCTOR_WITH_ANALYZER_H_
