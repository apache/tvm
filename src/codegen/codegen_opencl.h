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
 * \file codegen_opencl.h
 * \brief Generate OpenCL device code.
 */
#ifndef TVM_CODEGEN_CODEGEN_OPENCL_H_
#define TVM_CODEGEN_CODEGEN_OPENCL_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenOpenCL final : public CodeGenC {
 public:
  CodeGenOpenCL();
  void AddFunction(LoweredFunc f);
  std::string Finish();

  // override print thread tag.
  void InitFuncState(LoweredFunc f) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final; // NOLINT(*)
  void PrintStorageSync(const Call* op) final;  // NOLINT(*)
  void PrintType(Type t, std::ostream& os) final; // NOLINT(*)
  std::string GetVecLoad(Type t, const Variable* buffer,
                         Expr base) final;
  void PrintVecStore(const Variable* buffer,
                     Type t, Expr base,
                     const std::string& value) final;  // NOLINT(*)
  // the address of load/store
  void PrintVecAddr(const Variable* buffer, Type t,
                    Expr base, std::ostream& os);  // NOLINT(*)
  std::string CastFromTo(std::string value, Type from, Type target); // NOLINT(*)

  // overload visitor
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const Call* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const Select* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const FloatImm *op, std::ostream& os) final; // NOLINT(*)

 private:
  // whether enable fp16 and fp64 extension
  bool enable_fp16_{false};
  bool enable_fp64_{false};
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_OPENCL_H_
