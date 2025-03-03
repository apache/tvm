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
 * This module generates WGSL shading language.
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
 * We only leverage the C for expression generation and
 * write most of the language generations.
 */
class CodeGenWebGPU final : public CodeGenC {
 public:
  explicit CodeGenWebGPU(Target target);
  // overrides
  std::string Finish() final;
  using CodeGenC::AddFunction;
  runtime::FunctionInfo AddFunction(const PrimFunc& f, bool skip_readonly_decl);  // NOLINT(*)
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
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;      // NOLINT(*)
  void VisitExpr_(const LetNode* op, std::ostream& os) final;         // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;    // NOLINT(*)
  void VisitExpr_(const IntImmNode* op, std::ostream& os) final;      // NOLINT(*)

  // stmt printing
  void VisitStmt_(const LetStmtNode* op) final;
  void VisitStmt_(const BufferStoreNode* op) final;
  void VisitStmt_(const ForNode* op) final;
  void VisitStmt_(const AllocateNode* op) final;
  void VisitStmt_(const AssertStmtNode* op) final;
  void VisitStmt_(const AllocateConstNode* op) final;
  void VisitStmt_(const WhileNode* op) final;

 private:
  /*!
   * \brief Enforce value to be U32.
   */
  static PrimExpr EnforceU32(PrimExpr value);
  /*!
   * \brief Storage type of bool values.
   */
  DataType boolean_storage_type_{DataType::Int(8)};

  // whether enable fp16
  bool enable_fp16_{false};

  /*! \brief the header stream for function label and enable directive if any, goes before any other
   * declaration */
  std::ostringstream header_stream;

  Target target_;
};
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_WEBGPU_H_
