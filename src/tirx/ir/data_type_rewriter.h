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
#ifndef TVM_TIR_IR_DATA_TYPE_REWRITER_H_
#define TVM_TIR_IR_DATA_TYPE_REWRITER_H_

#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>

namespace tvm {
namespace tirx {

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
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

 protected:
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::RampNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SubNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MulNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::DivNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::ModNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MinNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::NENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) override;

  /*! \brief Whether to clamp shift amounts after narrowing signed integers. */
  virtual bool ShouldClampShiftAmounts() const { return false; }

  // a map from IterVar before rewrite to that after rewrite,
  // ensures one old IterVar maps to exactly one new IterVar
  std::unordered_map<const IterVarNode*, IterVar> ivmap_;
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
 public:
  using DataTypeLegalizer::Mutate;
  using DataTypeLegalizer::Mutate_;

 protected:
  using Parent = DataTypeLegalizer;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) override;
  ffi::Array<PrimExpr> VisitIndices(const ffi::Array<PrimExpr>& indices, InplaceMode inplace_mode);
  UnchangedOr<Stmt> Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::NENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) override;

  ffi::Map<ffi::String, ffi::Any> VisitBlockAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations);
  BufferRegion VisitBufferRegion(const BufferRegion& region);
  IterVar VisitIterVar(const IterVar& iter_var);
  // indicator of index expr to rewrite
  bool is_enabled_{false};
  // indicator of condition
  bool is_condition_{false};
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
  using IndexDataTypeRewriter::Mutate;
  using IndexDataTypeRewriter::Mutate_;
  explicit IndexDataTypeNormalizer(PrimType target_data_type);
  PrimFunc Rewrite(PrimFunc func);

 protected:
  using Parent = IndexDataTypeRewriter;

  UnchangedOr<PrimExpr> Mutate_(const IntImmNode* op, InplaceMode inplace_mode) override;

  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* op, InplaceMode inplace_mode) override;

  /*! \brief Specifies which data type we can rewrite */
  virtual bool CanRewriteDType(PrimType dtype) const;

  PrimType target_data_type_ = PrimType::Int(64);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIR_IR_DATA_TYPE_REWRITER_H_
