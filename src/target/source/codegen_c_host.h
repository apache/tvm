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
#include <unordered_set>
#include <utility>
#include <vector>

#include "codegen_c.h"
#include "tvm/target/codegen.h"
#include "tvm/tir/expr.h"

namespace tvm {
namespace codegen {

class CodeGenCHost : public CodeGenC {
 public:
  CodeGenCHost();
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl, std::string target_str,
            const std::unordered_set<std::string>& devices);

  void InitGlobalContext();
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f) override;
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f, bool emit_fwd_func_decl);
  /*!
   * \brief Add functions from the (unordered) range to the current module in a deterministic
   * order. This helps with debugging.
   *
   * \param functions A vector of unordered range of current module.
   */
  void AddFunctionsOrdered(std::vector<std::pair<tvm::GlobalVar, tvm::BaseFunc>> functions);
  void DefineModuleName();

  using CodeGenC::PrintType;
  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)
  void PrintFuncPrefix(std::ostream& os) final;        // NOLINT(*)

  // overload visitor functions
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) override;    // NOLINT(*)
  // overload min and max to use the ternary operator, so we don't rely on the
  // standard library implementations
  void VisitExpr_(const MinNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;  // NOLINT(*)

  void VisitStmt_(const AssertStmtNode* op) final;  // NOLINT(*)

  void GenerateForwardFunctionDeclarations(String global_symbol, const Array<Type>& arg_types,
                                           const Type& ret_type) override;
  Array<String> GetFunctionNames() { return function_names_; }

 private:
  /* \brief Internal structure to store information about function calls */
  struct FunctionInfo {
    /* \brief function name */
    std::string func_name;
    /* number of arguments required by the function */
    int64_t num_args;
    /* \brief name of resource_handle to pass */
    std::string resource_handle_name;
  };
  std::string module_name_;
  /* \brief mapping global packed func to the unique name */
  std::unordered_map<std::string, std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;
  /*! \brief whether to emit forwared function declarations in the resulting C code */
  bool emit_fwd_func_decl_;

  FunctionInfo GetFunctionInfo(const CallNode* op, bool has_resource_handle);
  std::string GetPackedName(const CallNode* op);
  void PrintGetFuncFromBackend(const std::string& func_name, const std::string& packed_func_name);
  void PrintFuncCall(const std::string& packed_func_name, int num_args);
  void PrintFuncCallC(const std::string& packed_func_name, int num_args,
                      const std::string& resource_handle_name);

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
