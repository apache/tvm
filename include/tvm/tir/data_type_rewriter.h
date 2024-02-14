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
 * \file data_type_rewriter.h
 * \brief Rewrite the data type of expressions.
 */
#ifndef TVM_TIR_DATA_TYPE_REWRITER_H_
#define TVM_TIR_DATA_TYPE_REWRITER_H_

#include <tvm/tir/stmt_functor.h>

#include <unordered_map>

namespace tvm {
namespace tir {

/*!
 * \brief Legalize the data types of expressions to make sure they are consistent with other
 * parts of the program.
 *
 * It enforces the following rules:
 * - The data type of the index variable in a loop must be consistent with the data type of the loop
 *  bounds.
 * - The data type of the binary and ternary expressions must be consistent with the data types of
 * each of their operands.
 * - The data type of the bounds and binding values of block iter vars must be consistent with the
 * data type of the block iter vars.
 *
 * Usually we enforce the consistency of data types when constructing the IR nodes. However, such
 * inconsistency may happen as a result of IR mutation in some passes. This class can be used as
 * base class of such passes to ensure the consistency of data types.
 */
class DataTypeLegalizer : public StmtExprMutator {
 protected:
  Stmt VisitStmt_(const ForNode* op) override;
  Stmt VisitStmt_(const AttrStmtNode* op) override;
  Stmt VisitStmt_(const BlockRealizeNode* op) override;
  Stmt VisitStmt_(const BlockNode* op) override;
  Stmt VisitStmt_(const LetStmtNode* op) override;
  PrimExpr VisitExpr_(const VarNode* op) override;
  PrimExpr VisitExpr_(const SelectNode* op) override;
  PrimExpr VisitExpr_(const RampNode* op) override;
  PrimExpr VisitExpr_(const AddNode* op) override;
  PrimExpr VisitExpr_(const SubNode* op) override;
  PrimExpr VisitExpr_(const MulNode* op) override;
  PrimExpr VisitExpr_(const DivNode* op) override;
  PrimExpr VisitExpr_(const ModNode* op) override;
  PrimExpr VisitExpr_(const FloorDivNode* op) override;
  PrimExpr VisitExpr_(const FloorModNode* op) override;
  PrimExpr VisitExpr_(const MinNode* op) override;
  PrimExpr VisitExpr_(const MaxNode* op) override;
  PrimExpr VisitExpr_(const EQNode* op) override;
  PrimExpr VisitExpr_(const NENode* op) override;
  PrimExpr VisitExpr_(const LTNode* op) override;
  PrimExpr VisitExpr_(const LENode* op) override;
  PrimExpr VisitExpr_(const GTNode* op) override;
  PrimExpr VisitExpr_(const GENode* op) override;
  PrimExpr VisitExpr_(const CallNode* op) override;
  PrimExpr VisitExpr_(const CastNode* op) override;
  PrimExpr VisitExpr_(const LetNode* op) override;

  using StmtExprMutator::VisitExpr_;
  using StmtExprMutator::VisitStmt_;

  // a map from IterVar before rewrite to that after rewrite,
  // ensures one old IterVar maps to exactly one new IterVar
  std::unordered_map<const IterVarNode*, IterVar> ivmap_;
  // a map from original vars to ones with new dtype
  std::unordered_map<const VarNode*, Var> var_remap_;
};

/*!
 * \brief Data type rewriter for buffer indices.
 *
 * Detect the components of buffer indices that should be considered for data type rewriting.
 * This class doesn't perform actual rewriting of data types. During recursive visiting, the
 * internal flags `is_enabled_` and `is_conditional_` are used to indicate whether the current
 * expression is a buffer index or a conditional expression, which can be used in the sub-classes to
 * implement different rewriting rules.
 */
class IndexDataTypeRewriter : public DataTypeLegalizer {
 protected:
  using Parent = DataTypeLegalizer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;

  Stmt VisitStmt_(const BlockRealizeNode* op) override;
  Stmt VisitStmt_(const BlockNode* op) override;
  Stmt VisitStmt_(const BufferStoreNode* op) override;
  Stmt VisitStmt_(const AttrStmtNode* op) override;
  PrimExpr VisitExpr_(const BufferLoadNode* op) override;
  Array<PrimExpr> VisitIndices(Array<PrimExpr> indices);
  Stmt VisitStmt_(const IfThenElseNode* op) override;
  Stmt VisitStmt_(const DeclBufferNode* op) override;
  Stmt VisitStmt_(const AllocateNode* op) override;
  PrimExpr VisitExpr_(const EQNode* op) override;
  PrimExpr VisitExpr_(const NENode* op) override;
  PrimExpr VisitExpr_(const LTNode* op) override;
  PrimExpr VisitExpr_(const LENode* op) override;
  PrimExpr VisitExpr_(const GTNode* op) override;
  PrimExpr VisitExpr_(const GENode* op) override;
  PrimExpr VisitExpr_(const CallNode* op) override;
  PrimExpr VisitExpr_(const SelectNode* op) override;

  Stmt VisitStmt_(const ForNode* op) override;

  Buffer VisitBuffer(const Buffer& buffer);
  Buffer GetRemappedBuffer(const Buffer& buffer);
  Map<String, ObjectRef> VisitBlockAnnotations(const Map<String, ObjectRef>& annotations);
  BufferRegion VisitBufferRegion(const BufferRegion& region);
  IterVar VisitIterVar(const IterVar& iter_var);
  // indicator of index expr to rewrite
  bool is_enabled_{false};
  // indicator of condition
  bool is_condition_{false};

  Map<Buffer, Buffer> buffer_remap_;
};

/*!
 * \brief Normalize the data types of buffer shapes and indices to the same data type.
 *
 * This pass rewrites the data types of buffer shapes and indices to the specified data type. It
 * assumes the specified data type is large enough to hold the original ranges of buffer shapes and
 * indices.
 */
class IndexDataTypeNormalizer : public IndexDataTypeRewriter {
 public:
  explicit IndexDataTypeNormalizer(DataType target_data_type);
  PrimFunc Rewrite(PrimFunc func);

 protected:
  using Parent = IndexDataTypeRewriter;
  using Parent::VisitExpr_;
  using Parent::VisitStmt_;
  PrimExpr VisitExpr_(const IntImmNode* op) override;
  PrimExpr VisitExpr_(const VarNode* op) override;
  PrimExpr VisitExpr_(const CastNode* op) override;

  /*! \brief Specifies which data type we can rewrite */
  virtual bool CanRewriteDType(DataType dtype) const;

  DataType target_data_type_ = DataType::Int(64);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_DATA_TYPE_REWRITER_H_
