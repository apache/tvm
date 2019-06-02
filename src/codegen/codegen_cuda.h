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
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.h
 * \brief Utility to generate cuda code
 */
#ifndef TVM_CODEGEN_CODEGEN_CUDA_H_
#define TVM_CODEGEN_CODEGEN_CUDA_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCUDA final : public CodeGenC {
 public:
  CodeGenCUDA();
  void Init(bool output_ssa);
  void AddFunction(LoweredFunc f);
  std::string Finish();
  bool need_include_path() {
    return (enable_fp16_ || enable_int8_ || need_math_constants_h_);
  }
  // override behavior
  void VisitStmt_(const ir::For* op) final;
  void PrintStorageSync(const Call* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(
      const std::string&op, Type t,
      Expr lhs, Expr rhs, std::ostream& os) final;  // NOLINT(*)
  void PrintType(Type t, std::ostream& os) final; // NOLINT(*)
  void PrintVecElemLoad(
      const std::string& vec, Type t, int i, std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(
      const std::string& vec, Type t, int i, const std::string& value) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  // overload visitor
  void VisitExpr_(const Ramp* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const FloatImm *op, std::ostream& os) final;
  void VisitStmt_(const Evaluate *op) final;

 private:
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
  friend void PrintConst(const FloatImm* op, std::ostream& os, CodeGenCUDA* p);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_CUDA_H_
