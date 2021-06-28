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

#include <string>
#include <unordered_map>
#include <vector>

#include "codegen_c.h"
#include "tvm/target/codegen.h"
#include "tvm/tir/expr.h"

namespace tvm {
namespace codegen {

class CodeGenCHost final : public CodeGenC {
 public:
  CodeGenCHost();
  void Init(bool output_ssa, bool emit_asserts, std::string target_str);

  void AddFunction(const PrimFunc& f);

  void DefineModuleName();

  /*! \brief Add linked parameters, if they are present. */
  void DeclareParameters(Map<String, LinkedParam> params);
  void LinkParameters(Map<String, LinkedParam> params);

  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintFuncPrefix() final;                        // NOLINT(*)
  void PrintFinalReturn() final;                       // NOLINT(*)

  // overload visitor functions
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) final;       // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void VisitExpr_(const MinNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;  // NOLINT(*)

  void VisitStmt_(const AssertStmtNode* op) final;  // NOLINT(*)

  Array<String> GetFunctionNames() { return function_names_; }

 private:
  /* \brief Internal structure to store information about function calls */
  struct FunctionInfo {
    /* \brief function name */
    std::string func_name;
    /* packed name of the function */
    std::string func_name_packed;
    /* number of arguments required by the function */
    int64_t num_args;
  };
  std::string module_name_;
  /* \brief mapping global packed func to the unique name */
  std::unordered_map<std::string, std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;

  FunctionInfo GetFunctionInfo(const CallNode* op);
  void PrintGetFuncFromBackend(const std::string& func_name, const std::string& packed_func_name);
  void PrintFuncCall(const std::string& packed_func_name, int num_args);
  void PrintFuncCallC(const std::string& packed_func_name, int num_args);

  /*!
   * \brief Print ternary conditional operator implementing binary `op`
   * Forces the operands to be in SSA form.
   * \param op binary operator being expressed
   * \param compare string representation of comparison operator
   * \param os stream reference to print into
   */
  template <typename T>
  inline void PrintTernaryCondExpr(const T* op, const char* compare,
                                   std::ostream& os);  // NOLINT(*)
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_C_HOST_H_
