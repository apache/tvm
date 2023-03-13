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
#ifndef TVM_TIR_SCHEDULE_IR_COMPARATOR_H_
#define TVM_TIR_SCHEDULE_IR_COMPARATOR_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./utils.h"

namespace tvm {
namespace tir {

using ExprComparator = ExprFunctor<bool(const PrimExpr& n, const PrimExpr& other)>;
using StmtComparator = StmtFunctor<bool(const Stmt& n, const Stmt& other)>;

/*! \brief Deep comparison to check if two IR ASTs are equivalent for tensorization*/
class TensorizeComparator : public ExprComparator, public StmtComparator {
 public:
  /*!
   * \brief Constructor of TensorizeComparator
   * \param assert_mode Whether to raise an error if the two IR ASTs do not match.
   * \param lhs_mod The IRModule of the LHS. This is used for error reporting.
   */
  explicit TensorizeComparator(IRModule lhs_mod, bool assert_mode = true)
      : lhs_mod_(std::move(lhs_mod)), assert_mode_(assert_mode) {}

  bool VisitExpr(const PrimExpr& n, const PrimExpr& other) override;
  bool VisitStmt(const Stmt& n, const Stmt& other) override;

  bool VisitStmt_(const ForNode* op, const Stmt& other) override;
  bool VisitStmt_(const SeqStmtNode* op, const Stmt& other) override;
  bool VisitStmt_(const BufferStoreNode* op, const Stmt& other) override;
  bool VisitStmt_(const BlockRealizeNode* op, const Stmt& other) override;
  bool VisitStmt_(const BlockNode* op, const Stmt& other) override;

  bool VisitExpr_(const AddNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const SubNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const MulNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const DivNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const ModNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const EQNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const NENode* op, const PrimExpr& other) override;
  bool VisitExpr_(const LTNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const LENode* op, const PrimExpr& other) override;
  bool VisitExpr_(const GTNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const GENode* op, const PrimExpr& other) override;
  bool VisitExpr_(const AndNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const OrNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const MinNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const MaxNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const FloorDivNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const FloorModNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const IntImmNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const FloatImmNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const CastNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const VarNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) override;
  bool VisitExpr_(const SelectNode* op, const PrimExpr& other) override;

  /*! \brief Map from RHS buffer to LHS buffer */
  std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual> rhs_buffer_map_;
  /*! \brief Base indices of the LHS buffer. */
  std::unordered_map<Buffer, std::vector<PrimExpr>, ObjectPtrHash, ObjectPtrEqual> buffer_indices_;

 protected:
  bool DefEqual(const Var& lhs, const Var& rhs);
  virtual bool CompareBuffer(const Buffer& lhs, const Buffer& rhs);
  bool CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs);
  bool CompareAnnotation(const std::pair<String, ObjectRef>& lhs,
                         const std::pair<String, ObjectRef>& rhs);
  bool CompareAnnotationMap(const Map<String, ObjectRef>& lhs, const Map<String, ObjectRef>& rhs);
  template <typename T>
  bool CompareBufferAccess(const T* lhs, const T* rhs);
  template <typename T, typename Self, typename F>
  bool CompareArray(const Array<T>& lhs, const Array<T>& rhs, F Self::*cmp);
  bool CompareRange(const Range& lhs, const Range& rhs);
  bool CompareIterVar(const IterVar& lhs, const IterVar& rhs);
  void EmitError(const std::string& error_message);

  /*! \brief IRModule of the LHS stmt. */
  IRModule lhs_mod_;
  /*! \brief Whether assertion mode is enabled. */
  bool assert_mode_;
  /*! \brief Whether it is visiting the scope block (the outermost block). */
  bool is_scope_block = true;
  /*! \brief The arithmetic analyzer for comparing LHS and RHS */
  arith::Analyzer analyzer_;
  /*!
   * \brief The arithmetic analyzer for simplifying expressions on LHS.
   *  This analyzer only contains the domains of the iterators on LHS.
   */
  arith::Analyzer lhs_analyzer_;
  /*! \brief Additional error messages. Only used when assert_mode is true. */
  std::vector<std::string> error_messages_;
  // variable remap if any
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> equal_map_;
};

/*!
 * \brief IR comparator for auto tensorization.
 * This comparator is used to extract correspondence between the IR of the workload (LHS) and the
 * tensor intrin (RHS). Unlike `TensorizeComparator`, this comparator has relaxed requirements
 * during comparison. It ignores the loop structure (number of loops and their extents) and buffer
 * indices. It only requires the LHS and the RHS to have the same arithmetic operations and the same
 * dtype. With such relaxed requirements, workloads that can only match the tensor intrin after
 * certain transformations (e.g. im2col for conv2d) are allowed for auto tensorization.
 */
class AutoTensorizeComparator : public TensorizeComparator {
 public:
  explicit AutoTensorizeComparator(const IRModule& lhs_mod)
      : TensorizeComparator(lhs_mod, /* assert_mode=*/false) {}

 private:
  bool VisitExprDefault_(const Object* op, const PrimExpr& other) override;
  bool VisitStmtDefault_(const Object* op, const Stmt& other) override;

  bool VisitStmt_(const BlockNode* op, const Stmt& other) override;
  bool VisitStmt_(const BufferStoreNode* op, const Stmt& other) override;

  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) override;

  bool CompareBuffer(const Buffer& lhs, const Buffer& rhs) override;
  template <typename T>
  bool CompareBufferAccess(const T* lhs, const T* rhs);

 public:
  // Additional information extracted from LHS (the workload) and RHS (the tensor intrin).

  /*! \brief Block iters in the LHS stmt. */
  std::vector<IterVar> lhs_iters_;
  /*! \brief Block iters in the RHS stmt. */
  std::vector<IterVar> rhs_iters_;
  /*! \brief The buffer and its access indices in the LHS stmt. */
  std::unordered_map<Buffer, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>
      lhs_buffer_indices_map_;
  /*! \brief The buffer and its access indices in the RHS stmt. */
  std::unordered_map<Buffer, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>
      rhs_buffer_indices_map_;
  /*! \brief Map from LHS buffer to RHS buffer */
  std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual> lhs_buffer_map_;

 private:
  /*! \brief The domain of the inner block iters. */
  Map<Var, arith::IntSet> inner_iter_dom_map_;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_IR_COMPARATOR_H_
