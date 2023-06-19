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
 * \file tvm/src/tir/analysis/var_use_def_analyzer.h
 * \brief Variable definition and usage analysis class.
 */
#ifndef TVM_TIR_ANALYSIS_VAR_USE_DEF_ANALYSIS_H_
#define TVM_TIR_ANALYSIS_VAR_USE_DEF_ANALYSIS_H_

#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>

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
  Array<Buffer> undefined_buffers_;

  std::unordered_map<const VarNode*, int> use_count_;
  std::unordered_map<const VarNode*, int> def_count_;
  std::unordered_map<const BufferNode*, int> buffer_use_count_;
  std::unordered_map<const BufferNode*, int> buffer_def_count_;

 private:
  ExprDeepEqual deep_equal_;
  std::unordered_map<const VarNode*, const LetNode*> let_binding_;
  void VisitStmt_(const AttrStmtNode* op) final;

  void VisitStmt_(const LetStmtNode* op) final;

  void VisitStmt_(const ForNode* op) final;

  void VisitStmt_(const DeclBufferNode* op) final;

  void VisitStmt_(const AllocateNode* op) final;

  void VisitStmt_(const AllocateConstNode* op) final;

  void VisitStmt_(const BufferStoreNode* op) final;

  void VisitExpr_(const LetNode* op) final;

  void VisitExpr_(const VarNode* op) final;

  void VisitExpr_(const ReduceNode* op) final;

  void VisitExpr_(const BufferLoadNode* op) final;

  void HandleDef(const Var& v);
  void HandleUse(const Var& v);

  void HandleDef(const Buffer& buf);
  void HandleUse(const Buffer& buf);

  void VisitBuffer(const Buffer& buffer);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_ANALYSIS_VAR_USE_DEF_ANALYSIS_H_
