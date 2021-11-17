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
 * \file feature_visitor.h
 * \brief Base class for feature extractor.
 *        These features are used for machine learning cost model
 */

#ifndef TVM_AUTOTVM_FEATURE_VISITOR_H_
#define TVM_AUTOTVM_FEATURE_VISITOR_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <string>

namespace tvm {
namespace autotvm {

using namespace tvm::tir;

/*!
 * \brief Type of for loop, used as one-hot encoding in features
 */
enum AnnotationType {
  kBlockX,
  kBlockY,
  kBlockZ,
  kThreadX,
  kThreadY,
  kThreadZ,
  kUnrolled,
  kVectorized,
  kParallel,
  kSerial,
  kVirtualThread,
  kNum,
};

/*!
 * \brief A base class for feature extractor, used for processing
 * for loop and memory access in the IR
 */
class FeatureVisitor : public StmtExprVisitor {
 public:
  // for loop
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;

  // memory access
  void VisitExpr_(const BufferLoadNode* op) final;
  void VisitStmt_(const BufferStoreNode* op) final;

  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;

 protected:
  /*!
   * \brief Enter a for loop node
   * \param var The expression to be printed.
   * \param length The output stream
   * \param ann_type The type for the for loop
   * \return skip Whether skip this node
   */
  virtual bool EnterItervar_(tir::Var var, int64_t length, AnnotationType ann_type) = 0;
  /*! \brief Exit a for loop subtree */
  virtual void ExitItervar_() = 0;
  /*!
   * \brief Enter a memory access node
   * \param buffer_var The buffer to access.
   * \param index Index expression
   */
  virtual void EnterMem_(tir::Var buffer_var, tvm::PrimExpr index) = 0;
  /*! \brief Exit a memory access node */
  virtual void ExitMem_() = 0;
};

}  // namespace autotvm
}  // namespace tvm

#endif  // TVM_AUTOTVM_FEATURE_VISITOR_H_
