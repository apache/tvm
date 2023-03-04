
/*
 * licensed to the apache software foundation (asf) under one
 * or more contributor license agreements.  see the notice file
 * distributed with this work for additional information
 * regarding copyright ownership.  the asf licenses this file
 * to you under the apache license, version 2.0 (the
 * "license"); you may not use this file except in compliance
 * with the license.  you may obtain a copy of the license at
 *
 *   http://www.apache.org/licenses/license-2.0
 *
 * unless required by applicable law or agreed to in writing,
 * software distributed under the license is distributed on an
 * "as is" basis, without warranties or conditions of any
 * kind, either express or implied.  see the license for the
 * specific language governing permissions and limitations
 * under the license.
 */

/*!
 * \file tvm/src/tir/analysis/var_use_def_analyzer.h
 * \brief Variable definition and usage analysis class.
 */
#ifndef TVM_TIR_ANALYSIS_VAR_USE_DEF_ANALZER_H_
#define TVM_TIR_ANALYSIS_VAR_USE_DEF_ANALZER_H_

#include <tvm/tir/analysis.h>

#include "../../runtime/thread_storage_scope.h"
#include "../transforms/ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Visitor class to perform use/def analysis, also delete unreferenced lets.
 * \param defined_vars Variables that have been defined.
 * \param visit_thread_extent Whether enters thread extent expressions or not.
 * \sa UndefinedVars
 */
class VarUseDefAnalyzer : public StmtExprVisitor {
 public:
  explicit VarUseDefAnalyzer(const Array<Var>& defined_vars, bool visit_thread_extent = true);
  // The fields are publically readible to
  // be accessible to the users.
  bool visit_thread_extent_{true};
  Array<Var> undefined_;

  std::unordered_map<const VarNode*, int> use_count_;
  std::unordered_map<const VarNode*, int> def_count_;

 private:
  ExprDeepEqual deep_equal_;
  std::unordered_map<Var, const LetNode*, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  void VisitStmt_(const AttrStmtNode* op) final;

  void VisitStmt_(const LetStmtNode* op) final;

  void VisitStmt_(const ForNode* op) final;

  void VisitStmt_(const AllocateNode* op) final;

  void VisitStmt_(const AllocateConstNode* op) final;

  void VisitStmt_(const StoreNode* op) final;

  void VisitStmt_(const BufferStoreNode* op) final;

  void VisitExpr_(const LetNode* op) final;

  void VisitExpr_(const VarNode* op) final;

  void VisitExpr_(const ReduceNode* op) final;

  void VisitExpr_(const LoadNode* op) final;

  void VisitExpr_(const BufferLoadNode* op) final;

  void HandleDef(const VarNode* v);

  void HandleUse(const VarNode* v);

  void VisitBuffer(Buffer buffer);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_ANALYSIS_VAR_USE_DEF_ANALZER_H_
