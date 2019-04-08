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
 *  Copyright (c) 2018 by Contributors
 * \file ir_builder.h
 * \brief Utility for building SPIRV code block
 */
#ifndef TVM_CODEGEN_SPIRV_CODEGEN_SPIRV_H_
#define TVM_CODEGEN_SPIRV_CODEGEN_SPIRV_H_

#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/lowered_func.h>

#include <vector>
#include <memory>
#include <unordered_map>

#include "ir_builder.h"
#include "../../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

using namespace ir;

/*!
 * \brief Code generator into SPIRV
 */
class CodeGenSPIRV:
      public ExprFunctor<spirv::Value(const Expr&)>,
      public StmtFunctor<void(const Stmt&)> {
 public:
  /*!
   * \brief Compile and add function f to the current module.
   * \param f The function to be added.
   * \return The final spirv module.
   */
  virtual std::vector<uint32_t> BuildFunction(const LoweredFunc& f);
  /*!
   * \brief Create Value for expression e
   * \param e The expression to be created value for.
   * \return created value.
   */
  spirv::Value MakeValue(const Expr& e) {
    return VisitExpr(e);
  }
  // override codegen
  spirv::Value VisitExpr_(const Variable* op) override;
  spirv::Value VisitExpr_(const Cast* op) override;
  spirv::Value VisitExpr_(const IntImm* op) override;
  spirv::Value VisitExpr_(const UIntImm* op) override;
  spirv::Value VisitExpr_(const FloatImm* op) override;
  spirv::Value VisitExpr_(const StringImm* op) override;
  spirv::Value VisitExpr_(const Add* op) override;
  spirv::Value VisitExpr_(const Sub* op) override;
  spirv::Value VisitExpr_(const Mul* op) override;
  spirv::Value VisitExpr_(const Div* op) override;
  spirv::Value VisitExpr_(const Mod* op) override;
  spirv::Value VisitExpr_(const Min* op) override;
  spirv::Value VisitExpr_(const Max* op) override;
  spirv::Value VisitExpr_(const LT* op) override;
  spirv::Value VisitExpr_(const LE* op) override;
  spirv::Value VisitExpr_(const GT* op) override;
  spirv::Value VisitExpr_(const GE* op) override;
  spirv::Value VisitExpr_(const EQ* op) override;
  spirv::Value VisitExpr_(const NE* op) override;
  spirv::Value VisitExpr_(const And* op) override;
  spirv::Value VisitExpr_(const Or* op) override;
  spirv::Value VisitExpr_(const Not* op) override;
  spirv::Value VisitExpr_(const Select* op) override;
  spirv::Value VisitExpr_(const Let* op) override;
  spirv::Value VisitExpr_(const Call* op) override;
  spirv::Value VisitExpr_(const Ramp* op) override;
  spirv::Value VisitExpr_(const Broadcast* op) override;
  spirv::Value VisitExpr_(const Load* op) override;
  // stmt
  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const AssertStmt* op) override;
  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const Block* op) override;
  void VisitStmt_(const Evaluate* op) override;
  void VisitStmt_(const ProducerConsumer* op) override;

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
    Type content_type{Handle()};

    // Update content type if it hasn't beenupdated.
    void UpdateContentType(Type type) {
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
  spirv::Value GetThreadIndex(const IterVar& iv, const Expr& extent);
  spirv::Value CreateStorageSync(const Call* op);
  void Scalarize(const Expr& e,
                 std::function<void(int i, spirv::Value v)> f);
  // The builder
  std::unique_ptr<spirv::IRBuilder> builder_;
  // Work group size of three
  uint32_t workgroup_size_[3];
  // Likely branch
  uint32_t weight_likely_branch_{128};
  // the storage scope of allocation
  std::unordered_map<const Variable*, StorageInfo> storage_info_;
  // The definition of local variable.
  std::unordered_map<const Variable*, spirv::Value> var_map_;
  // The analyzer.
  std::unique_ptr<arith::Analyzer> analyzer_;
};

}  // namespace codegen
}  // namespace tvm


#endif  // TVM_CODEGEN_SPIRV_CODEGEN_SPIRV_H_
