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
 * \file ir_builder.h
 * \brief Utility for building SPIRV code block
 */
#ifndef TVM_TARGET_SPIRV_CODEGEN_SPIRV_H_
#define TVM_TARGET_SPIRV_CODEGEN_SPIRV_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/function.h>

#include <vector>
#include <memory>
#include <unordered_map>

#include "ir_builder.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

using namespace tir;

/*!
 * \brief Code generator into SPIRV
 */
class CodeGenSPIRV:
      public ExprFunctor<spirv::Value(const PrimExpr&)>,
      public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Compile and add function f to the current module.
   * \param f The function to be added.
   * \return The final spirv module.
   */
  virtual std::vector<uint32_t> BuildFunction(const PrimFunc& f);
  /*!
   * \brief Create Value for expression e
   * \param e The expression to be created value for.
   * \return created value.
   */
  spirv::Value MakeValue(const PrimExpr& e) {
    return VisitExpr(e);
  }
  // override codegen
  spirv::Value VisitExpr_(const VarNode* op) override;
  spirv::Value VisitExpr_(const CastNode* op) override;
  spirv::Value VisitExpr_(const IntImmNode* op) override;
  spirv::Value VisitExpr_(const FloatImmNode* op) override;
  spirv::Value VisitExpr_(const StringImmNode* op) override;
  spirv::Value VisitExpr_(const AddNode* op) override;
  spirv::Value VisitExpr_(const SubNode* op) override;
  spirv::Value VisitExpr_(const MulNode* op) override;
  spirv::Value VisitExpr_(const DivNode* op) override;
  spirv::Value VisitExpr_(const ModNode* op) override;
  spirv::Value VisitExpr_(const MinNode* op) override;
  spirv::Value VisitExpr_(const MaxNode* op) override;
  spirv::Value VisitExpr_(const LTNode* op) override;
  spirv::Value VisitExpr_(const LENode* op) override;
  spirv::Value VisitExpr_(const GTNode* op) override;
  spirv::Value VisitExpr_(const GENode* op) override;
  spirv::Value VisitExpr_(const EQNode* op) override;
  spirv::Value VisitExpr_(const NENode* op) override;
  spirv::Value VisitExpr_(const AndNode* op) override;
  spirv::Value VisitExpr_(const OrNode* op) override;
  spirv::Value VisitExpr_(const NotNode* op) override;
  spirv::Value VisitExpr_(const SelectNode* op) override;
  spirv::Value VisitExpr_(const LetNode* op) override;
  spirv::Value VisitExpr_(const CallNode* op) override;
  spirv::Value VisitExpr_(const RampNode* op) override;
  spirv::Value VisitExpr_(const BroadcastNode* op) override;
  spirv::Value VisitExpr_(const LoadNode* op) override;
  // stmt
  void VisitStmt_(const StoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;

 protected:
  /*! \brief The storage information */
  struct StorageInfo {
    /*! \brief The storage scope */
    runtime::StorageScope scope;
    /*! \brief Whether it is volatile */
    bool is_volatile{false};
    /*! \brief Whether it is volatile */
    bool content_fixed{false};
    /*! \brief Current content type */
    DataType content_type{DataType::Handle()};

    // Update content type if it hasn't beenupdated.
    void UpdateContentType(DataType type) {
      if (content_fixed) {
        CHECK_EQ(type, content_type)
            << "Cannot use two different content type in GLSL model";
      } else {
        this->content_type = type;
        content_fixed = true;
      }
    }
  };
  // Reset the state so it works for a new function.
  void InitFuncState();
  // Get the thread index
  spirv::Value GetThreadIndex(const IterVar& iv, const PrimExpr& extent);
  spirv::Value CreateStorageSync(const CallNode* op);
  void Scalarize(const PrimExpr& e,
                 std::function<void(int i, spirv::Value v)> f);
  // The builder
  std::unique_ptr<spirv::IRBuilder> builder_;
  // Work group size of three
  uint32_t workgroup_size_[3];
  // Likely branch
  uint32_t weight_likely_branch_{128};
  // the storage scope of allocation
  std::unordered_map<const VarNode*, StorageInfo> storage_info_;
  // The definition of local variable.
  std::unordered_map<const VarNode*, spirv::Value> var_map_;
  // The analyzer.
  std::unique_ptr<arith::Analyzer> analyzer_;
};

}  // namespace codegen
}  // namespace tvm


#endif  // TVM_TARGET_SPIRV_CODEGEN_SPIRV_H_
