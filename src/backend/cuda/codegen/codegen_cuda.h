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
 * \brief Utility to generate CUDA code
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_CUDA_H_
#define TVM_TARGET_SOURCE_CODEGEN_CUDA_H_

#include <tvm/target/codegen.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <string>
#include <unordered_map>

#include "../../../target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCUDA final : public CodeGenC {
 public:
  CodeGenCUDA(Target target);
  void Init(bool output_ssa);
  std::string Finish();
  bool need_include_path() {
    std::vector<std::string> tag_list{"fp16", "bf16", "int8",           "fp8",
                                      "fp6",  "fp4",  "math_constants", "mma"};
    return std::any_of(tag_list.begin(), tag_list.end(), [this](const std::string& tag) {
      return codegen_tags_.find(tag) != codegen_tags_.end();
    });
  }
  // override behavior
  void PrintFunctionSignature(const ffi::String& function_name, const PrimFunc& func,
                              std::ostream& os) final;
  void PrintExtraAttrs(const PrimFunc& f, std::ostream& os) final;  // NOLINT(*)
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const WhileNode* op) final;
  void PrintStorageSync(const CallNode* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(const std::string& op, DLDataType t, PrimExpr lhs, PrimExpr rhs,
                        std::ostream& os) final;         // NOLINT(*)
  void PrintType(DLDataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintVecConstructor(DLDataType t, std::ostream& os) final;
  void PrintVecElemLoad(const std::string& vec, DLDataType t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, DLDataType t, int i,
                         const std::string& value) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintVecElemLoadExpr(DLDataType t, int i, const std::string& value, std::ostream& os) final;
  std::string CastFromTo(std::string value, DLDataType from, DLDataType target) final;
  void AddUtilFunction(const std::string& name, const std::string& code);
  // overload visitor
  void VisitExpr_(const RampNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;
  void VisitExpr_(const CallNode* op, std::ostream& os) final;
  void VisitExpr_(const CastNode* op, std::ostream& os) final;
  void VisitStmt_(const EvaluateNode* op) final;
  void VisitStmt_(const AllocBufferNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;

  // Target
  Target target;

 protected:
  void PrintCallExtern(Type ret_type, ffi::String global_symbol, const ffi::Array<PrimExpr>& args,
                       bool skip_first_arg, std::ostream& os) final;  // NOLINT(*)

 private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string& value, const BufferLoadNode* op,
                           std::ostream& os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;

  // Whether clusterCtaIdx.x can be emitted as the linear cluster CTA rank.
  // This is only semantics-preserving for effectively 1-D clusters where the
  // y/z cluster-CTA extents are both one.
  bool cluster_cta_x_is_linear_rank_{false};

  // Codegen tags
  std::unordered_set<std::string> codegen_tags_;

  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ = Op::GetAttrMap<bool>("cuda.need_warp_shuffle");

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  std::unordered_map<int, int> barrier_count_;
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;
  // Functions to be added to the util functions during codegen
  std::unordered_map<std::string, std::string> util_funcs_;

  // The name prefix of the cuda::barrier array in shared memory
  const std::string cuda_barrier_name_ = "cubar";
  // The name prefix of the cuda::barrier::arrival_token array in registers
  const std::string cuda_barrier_arrival_token_name_ = "cubar_tok";

  std::unordered_map<const VarNode*, std::string> fragment_shapes;
  std::unordered_map<const VarNode*, std::string> fragment_layouts;
  friend void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenCUDA* p);
  void PrintWmmaScope(const std::string& scope, DLDataType t, const VarNode* variable,
                      std::ostream& os);
  int32_t GetWmmaFragmentSize(const std::string& scope, const VarNode* variable, int32_t size);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_CUDA_CODEGEN_CUDA_H_
