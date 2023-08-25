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
 * \file codegen_webgpu.h
 * \brief Generate WebGPU shaders in WGSL.
 *
 * This module generates WGSL shading langauge.
 * See https://www.w3.org/TR/WGSL/ for the language reference.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_WEBGPU_H_
#define TVM_TARGET_SOURCE_CODEGEN_WEBGPU_H_

#include <tvm/target/codegen.h>

#include <string>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

/*!
 * \brief WebGPU code generator.
 *
 * Note WGSL have a different syntax from normal C.
 * We only leevrage the C for expression generation and
 * write most of the language generations.
 */
class CodeGenWebGPU final : public CodeGenC {
 public:
  explicit CodeGenWebGPU(Target target);
  // overrides
  std::string Finish() final;
  void PrintFunctionSignature(const String& function_name, const PrimFunc& func,
                              std::ostream& os) final;
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f) final;
  void InitFuncState(const PrimFunc& f) final;
  void PrintStorageSync(const CallNode* op) final;     // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;       // NOLINT(*)

  // assignment printing
  void PrintSSAAssign(const std::string& target, const std::string& src, DataType type) final;

  // overload visitor
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;   // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) override;   // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os) final;      // NOLINT(*)

  // stmt printing
  void VisitStmt_(const LetStmtNode* op) final;
  void VisitStmt_(const BufferStoreNode* op) final;
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const AllocateNode* op) final;
  void VisitStmt_(const AttrStmtNode* op) final;
  void VisitStmt_(const AssertStmtNode* op) final;
  void VisitStmt_(const AllocateConstNode* op) final;

 private:
  /*!
   * \brief Records the workgroup size of the kernel.
   */
  uint32_t workgroup_size_[3];
  /*!
   * \brief Storage type of bool values.
   */
  DataType boolean_storage_type_{DataType::Int(8)};
  Target target_;
};
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_WEBGPU_H_
