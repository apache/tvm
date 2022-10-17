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
 * \file codegen_opencl.h
 * \brief Generate OpenCL device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_OPENCL_H_
#define TVM_TARGET_SOURCE_CODEGEN_OPENCL_H_

#include <tvm/target/codegen.h>

#include <string>
#include <unordered_map>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenOpenCL final : public CodeGenC {
 public:
  CodeGenOpenCL();
  std::string Finish();

  // override print thread tag.
  void InitFuncState(const PrimFunc& f) final;
  void PrintFuncPrefix() final;                                              // NOLINT(*)
  void PreFunctionBody(const PrimFunc& f) final;                             // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;                             // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintStorageSync(const CallNode* op) final;                           // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;                        // NOLINT(*)
  void PrintType(const Type& type, std::ostream& os) final;                  // NOLINT(*)
  std::string GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) final;
  void PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                     const std::string& value) final;  // NOLINT(*)
  // the address of load/store
  void PrintVecAddr(const BufferNode* buffer, DataType t, PrimExpr base,
                    std::ostream& os);                                           // NOLINT(*)
  void PrintRestrict(const Var& v, std::ostream& os) final;                      // NOLINT(*)
  std::string CastFromTo(std::string value, DataType from, DataType target);     // NOLINT(*)
  std::string CastTo(std::string value, DataType target);                        // NOLINT(*)
  void SetTextureScope(const std::unordered_map<const VarNode*, std::string>&);  // NOLINT(*)

  // overload visitor
  void VisitStmt_(const AllocateNode* op) final;                     // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const CastNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;   // NOLINT(*)
  void VisitStmt_(const StoreNode* op) final;                        // NOLINT(*)
  void VisitStmt_(const BufferStoreNode* op) final;                  // NOLINT(*)

  // overload min and max to avoid ambiguous call errors
  void VisitExpr_(const MinNode* op, std::ostream& os) final;
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;
  void VisitExpr_(const AndNode* op, std::ostream& os) final;
  void VisitExpr_(const OrNode* op, std::ostream& os) final;
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;

 private:
  // whether enable fp16 and fp64 extension
  bool enable_fp16_{false};
  bool enable_fp64_{false};
  // Whether to enable atomics extension.
  bool enable_atomics_{false};
  // Whether to enable sampler or sampler-less texture reads,
  // where the choice depends on the OpenCL version used.
  bool enable_compliant_texture_reads_{false};
  // Key to disable use of texture SSA in certain scenarios. For example,
  // when loaded value is stored directly to a user declared l-value buffer
  bool need_texture_ssa_{true};
  // Mapping from buffer to allocation size.
  // Useful to track when a scalar store of a vectorized texture load is required.
  std::unordered_map<const Object*, size_t> allocation_size_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_OPENCL_H_
