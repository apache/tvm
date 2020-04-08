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
 * \file codegen_c_host.h
 * \brief Generate C host code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_C_HOST_H_
#define TVM_TARGET_SOURCE_CODEGEN_C_HOST_H_

#include <set>
#include <string>
#include "tvm/target/codegen.h"
#include "tvm/tir/expr.h"
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCHost final : public CodeGenC {
 public:
  CodeGenCHost();
  void Init(bool output_ssa, bool emit_asserts);

  void PrintType(DataType t, std::ostream& os) final; // NOLINT(*)
  void PrintFuncPrefix() final; // NOLINT(*)
  void PrintFinalReturn() final; // NOLINT(*)

  // overload visitor functions
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream& os) final; // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void VisitExpr_(const MinNode *op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const MaxNode *op, std::ostream& os) final;  // NOLINT(*)

  void VisitStmt_(const AssertStmtNode *op) final; // NOLINT(*)

 private:
  std::string module_name_;
  /* \brief tracks declared global variables which live despite GetUniqueName */
  std::set<std::string> declared_globals_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;

  void PrintGetFuncFromBackend(const std::string& func_name, const std::string& packed_func_name);
  void PrintFuncCall(const std::string& packed_func_name, int num_args);

  /*!
   * \brief Print ternary conditional operator implementing binary `op`
   * Forces the operands to be in SSA form.
   * \param op binary operator being expressed
   * \param compare string representation of comparison operator
   * \param os stream reference to print into
   */
  template <typename T>
  inline void PrintTernaryCondExpr(const T* op,
                                   const char* compare,
                                   std::ostream& os);  // NOLINT(*)
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_C_HOST_H_
