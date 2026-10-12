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

#ifndef TVM_S_TIR_IR_DATA_TYPE_REWRITER_H_
#define TVM_S_TIR_IR_DATA_TYPE_REWRITER_H_

#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <utility>

namespace tvm {
namespace s_tir {

/*!
 * \brief Legalize the data types of expressions to make sure they are consistent with other
 * parts of the program.
 *
 * It enforces the following rules:
 * - The data type of the index variable in a loop must be consistent with the data type of the loop
 *  bounds.
 * - The data type of the binary and ternary expressions must be consistent with the data types of
 * each of their operands.
 *
 * Usually we enforce the consistency of data types when constructing the IR nodes. However, such
 * inconsistency may happen as a result of IR mutation in some passes. This class can be used as
 * base class of such passes to ensure the consistency of data types.
 */
class DataTypeLegalizer : public tirx::StmtExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(DataTypeLegalizer, tirx::StmtExprMutator)
  using tirx::StmtExprMutator::Mutate;
  using tirx::StmtExprMutator::Mutate_;

 protected:
  explicit DataTypeLegalizer(const VTable* vtable) : tirx::StmtExprMutator(vtable) {}
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::RampNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LShiftNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::RShiftNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotNode* op, InplaceMode inplace_mode) override;
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
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) override;

  /*! \brief Whether to clamp shift amounts after narrowing signed integers. */
  virtual bool ShouldClampShiftAmounts() const { return false; }
};

/*!
 * \brief Data type rewriter for tensor indices.
 *
 * Detect the components of tensor indices that should be considered for data type rewriting.
 * This class doesn't perform actual rewriting of data types. During recursive visiting, the
 * internal flags `is_enabled_` and `is_conditional_` are used to indicate whether the current
 * expression is a tensor index or a conditional expression, which can be used in the sub-classes to
 * implement different rewriting rules.
 */
class IndexDataTypeRewriter : public DataTypeLegalizer {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(IndexDataTypeRewriter, DataTypeLegalizer)
  using DataTypeLegalizer::Mutate;
  using DataTypeLegalizer::Mutate_;

 protected:
  explicit IndexDataTypeRewriter(const VTable* vtable) : DataTypeLegalizer(vtable) {}
  using Parent = DataTypeLegalizer;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) override;
  ffi::Array<PrimExpr> VisitIndices(const ffi::Array<PrimExpr>& indices, InplaceMode inplace_mode);
  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::NENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::LENode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GTNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::GENode* op, InplaceMode inplace_mode) override;

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) override;

  // indicator of index expr to rewrite
  bool is_enabled_{false};
  // indicator of condition
  bool is_condition_{false};
};

// Explicit normalization for schedulable TE functions and tensor intrinsic bodies.
// Ordinary TIRX dtype passes run after block lowering.
class IndexDataTypeNormalizer : public IndexDataTypeRewriter {
 public:
  using Parent = IndexDataTypeRewriter;
  using Parent::Mutate;
  using Parent::Mutate_;
  explicit IndexDataTypeNormalizer(PrimType target_data_type)
      : Parent(GlobalVTable()), target_data_type_(std::move(target_data_type)) {}
  tirx::Function Rewrite(tirx::Function func);

  virtual UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode);

 protected:
  UnchangedOr<PrimExpr> Mutate_(const IntImmNode* op, InplaceMode inplace_mode) override;
  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* op, InplaceMode inplace_mode) override;

  /*! \brief Specifies which data type we can rewrite */
  virtual bool CanRewriteDType(PrimType dtype) const;

  PrimType target_data_type_ = PrimType::Int(64);

  static void InitVTable(VTable* table) {
    Parent::InitVTable(table);
    SetDispatch<IndexDataTypeNormalizer, SBlockNode>(table);
    SetDispatch<IndexDataTypeNormalizer, SBlockRealizeNode>(table);
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }

 private:
  ffi::Map<ffi::String, ffi::Any> VisitBlockAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations);
  s_tir::IterVar VisitIterVar(const s_tir::IterVar& iter_var);
  TensorRegion VisitTensorRegion(const TensorRegion& tensor_region);
};

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_IR_DATA_TYPE_REWRITER_H_
