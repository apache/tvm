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
 * \file codegen_cuda.h
 * \brief Utility to generate cuda code
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_CUDA_H_
#define TVM_TARGET_SOURCE_CODEGEN_CUDA_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <string>
#include <unordered_map>
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCUDA final : public CodeGenC {
 public:
  CodeGenCUDA();
  void Init(bool output_ssa);
  std::string Finish();
  bool need_include_path() {
    return (enable_fp16_ || enable_int8_ || need_math_constants_h_ || need_mma_h_);
  }
  // override behavior
  void PrintFuncPrefix() final;
  void VisitStmt_(const ForNode* op) final;
  void PrintStorageSync(const CallNode* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(
      const std::string& op, DataType t,
      PrimExpr lhs, PrimExpr rhs, std::ostream& os) final;  // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final; // NOLINT(*)
  void PrintVecElemLoad(
      const std::string& vec, DataType t, int i, std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(
      const std::string& vec, DataType t, int i, const std::string& value) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) final;
  // overload visitor
  void VisitExpr_(const RampNode* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream& os) final;
  void VisitExpr_(const CallNode *op, std::ostream& os) final;
  void VisitExpr_(const CastNode* op, std::ostream& os) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;

 private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string& value, const LoadNode* op,
                           std::ostream& os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final {
    return false;
  }

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};

  std::unordered_map<const VarNode*, std::string> fragment_shapes;
  std::unordered_map<const VarNode*, std::string> fragment_layouts;
  friend void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenCUDA* p);
  void PrintWmmaScope(
      const std::string& scope, DataType t, const VarNode* variable, std::ostream& os);
  int32_t GetWmmaFragmentSize(
      const std::string &scope, const VarNode* variable, int32_t size);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_CUDA_H_
