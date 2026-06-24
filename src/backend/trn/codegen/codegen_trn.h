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
 * \file codegen_trn.h
 * \brief Generate Metal device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_TRN_H_
#define TVM_TARGET_SOURCE_CODEGEN_TRN_H_

#include <tvm/target/codegen.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../../../target/source/codegen_c.h"

namespace tvm {
namespace codegen {

struct NKIInstructionCtx {
  std::unordered_set<const VarNode*> tensorized_loop_vars;
  std::unordered_map<const VarNode*, ffi::String> loopvar2dim;
  bool is_matmul_input = false;
  int buffer_index = -1;
  int used_var_cnt = 0;
  PrimType dst_dtype = PrimType::Void();
  PrimExpr mask;
  bool tensorizing = false;
};

class CodeGenTrainium final : public CodeGenC {
 public:
  explicit CodeGenTrainium(Target target);
  using CodeGenC::VisitExpr_;
  using CodeGenC::VisitStmt_;
  // override print thread tag.
  void PrintArgUnionDecl();
  void AddFunction(const GlobalVar& gvar, const PrimFunc& func) final;
  void InitFuncState(const PrimFunc& f) final;
  std::string GetStorageScopeStr(const std::string& scope);           // NOLINT(*)
  void VisitExpr_(const VarNode* op, std::ostream& os) final;         // NOLINT(*)
  void PrintType(DLDataType t, std::ostream& os) final;               // NOLINT(*)
  void VisitStmt_(const AllocBufferNode* op) final;                   // NOLINT(*)
  void VisitStmt_(const AttrStmtNode* op) final;                      // NOLINT(*)
  void VisitStmt_(const ForNode* op) final;                           // NOLINT(*)
  void VisitStmt_(const BufferStoreNode* op) final;                   // NOLINT(*)=
  void VisitStmt_(const EvaluateNode* op) final;                      // NOLINT(*)
  std::string PrintIndices(const ffi::Array<PrimExpr>& indices);      // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const FloorDivNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const FloorModNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitStmt_(const DeclBufferNode* op) final;                    // NOLINT(*)
  void VisitStmt_(const IfThenElseNode* op) final;                    // NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os) final;         // NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os) final;          // NOLINT(*)

 private:
  Target target_;
  NKIInstructionCtx ctx_;
  std::unordered_map<std::string, std::string> opcode_map_;
  std::unordered_map<Buffer, std::string, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_idmap_;
  std::unordered_map<const VarNode*, std::string> data_buffer_idmap_;
  std::unordered_map<const VarNode*, Buffer> data_decl_buffer_map_;
  bool is_outermost_loop_ = true;
};
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_TRN_H_
