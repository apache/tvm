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
 * \file codegen_hip.h
 * \brief Utility to generate hip source code
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_HIP_H_
#define TVM_TARGET_SOURCE_CODEGEN_HIP_H_

#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>
#include <unordered_map>

#include "codegen_c.h"
#include "codegen_cuda.h"

namespace tvm {
namespace codegen {

class CodeGenHIP final : public CodeGenC {
 public:
  CodeGenHIP();
  void Init(bool output_ssa);
  std::string Finish();
  bool need_include_path() { return (need_math_constants_h_ || need_wmma_h_); }
  // override behavior
  void PrintFuncPrefix(std::ostream& os) final;
  void PrintExtraAttrs(const PrimFunc& f, std::ostream& os) final;  // NOLINT(*)
  void VisitStmt_(const ForNode* op) final;
  void PrintStorageSync(const CallNode* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                        std::ostream& os){
    cuda_codegen_.PrintVecBinaryOp(op, t, lhs, rhs, os);
  };                                                   // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                        std::ostream& os){
    return cuda_codegen_.PrintVecElemLoad(vec, t, i, os);
  };  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, DataType t, int i, const std::string& value) {return cuda_codegen_.PrintVecElemStore(vec, t, i, value);};
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) final;
  std::string CastFromTo(std::string value, DataType from, DataType target) {
    return cuda_codegen_.CastFromTo(value, from, target);
  };
  // overload visitor
  void VisitExpr_(const RampNode* op, std::ostream& os) {
    return cuda_codegen_.VisitExpr_(op, os);
  };                                                                 // NOLINT(*)
  void VisitExpr_(const ShuffleNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;
  void VisitExpr_(const CallNode* op, std::ostream& os) final;
  void VisitExpr_(const CastNode* op, std::ostream& os) final;
  void VisitStmt_(const EvaluateNode* op) final;
  void VisitStmt_(const AllocateNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;

 protected:
   // Handle volatile loads
  void HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                           std::ostream& os) final;
  void PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                       bool skip_first_arg, std::ostream& os) final;  // NOLINT(*)

 private:
  CodeGenCUDA cuda_codegen_;
  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;

  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need mfma.h
  bool need_wmma_h_{false};
  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable int8
  bool enable_int8_{false};
  std::unordered_map<const VarNode*, std::string> fragment_shapes;
  std::unordered_map<const VarNode*, std::string> fragment_layouts;
  friend void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenHIP* p);
  void PrintWmmaScope(const std::string& scope, DataType t, const VarNode* variable,
                      std::ostream& os);
  int32_t GetWmmaFragmentSize(const std::string& scope, const VarNode* variable, int32_t size);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_CUDA_H_
