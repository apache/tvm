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
 * \file codegen_c_host.h
 * \brief Generate C host code.
 */
#ifndef TVM_CODEGEN_CODEGEN_C_HOST_H_
#define TVM_CODEGEN_CODEGEN_C_HOST_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCHost final : public CodeGenC {
 public:
  CodeGenCHost();
  void Init(bool output_ssa);
  void AddFunction(LoweredFunc f);
  std::string Finish();

  void PrintType(Type t, std::ostream& os) final; // NOLINT(*)

  // overload visitor functions
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const Call *op, std::ostream& os) final; // NOLINT(*)
  void VisitStmt_(const AssertStmt *op) final; // NOLINT(*)

 private:
  std::string module_name;
  void PrintGetFuncFromBackend(std::string func_name, std::string packed_func_name);
  void PrintFuncCall(std::string packed_func_name, int num_args);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_C_HOST_H_
