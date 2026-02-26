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
 * \file tvm/tir/analysis.h
 * \brief Analysis utilities and passes for TIR.
 */
#ifndef TVM_TIR_ANALYSIS_H_
#define TVM_TIR_ANALYSIS_H_

#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

#include <optional>
#include <string>

namespace tvm {

namespace arith {
class Analyzer;
}

namespace tir {

/*!
 * \brief Compare two expressions recursively and check if they are equal
 *        to each other without var remapping.
 *
 *  This function does not remap variable bindings, it will not
 *  return true for (let x = 1 in x + 1) vs (let y = 1 in y + 1), unless x.same_as(y).
 *
 *  Use StructuralEqual for such cases.
 *
 *  Due to the restriction of not remapping variables, this function can run
 *  faster than StructuralEqual and can be used as a utility function during arithmetic
 *  simplifications.
 *
 * \sa StructuralEqual
 */
struct ExprDeepEqual {
 public:
  TVM_DLL bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const;
};

/*!
 * \brief Visit the PrimFuncs in the IRModule
 * \tparam FLambda The type of the PrimFunc visitor
 * \param mod The IRModule to be visited
 * \param fvisit The visitor to the PrimFuncs in the IRModule
 */
template <class FLambda>
inline void VisitPrimFuncs(const IRModule& mod, FLambda fvisit) {
  for (const auto& kv : mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (const auto* prim_func = base_func.as<PrimFuncNode>()) {
      fvisit(prim_func);
    }
  }
}

/*!
 * \brief Find undefined vars in the statement.
 * \param stmt The statement to be checked.
 * \param defs The vars that is defined.
 * \return Array of undefined vars.
 */
TVM_DLL ffi::Array<Var> UndefinedVars(const Stmt& stmt, const ffi::Array<Var>& defs);

/*!
 * \brief Find undefined vars in the expression.
 * \param expr The expression to be checked.
 * \return Array of undefined vars.
 */
TVM_DLL ffi::Array<Var> UndefinedVars(const PrimExpr& expr);

/*!
 * \brief Find undefined vars in the expression.
 * \param expr The expression to be checked.
 * \param defs The vars that is defined.
 * \return Array of undefined vars.
 */
TVM_DLL ffi::Array<Var> UndefinedVars(const PrimExpr& expr, const ffi::Array<Var>& defs);

/*!
 * \brief Analyze the side effect of an expression
 * \param expr The expression to be checked.
 *
 * \return CallEffectKind, can be kPure, kReadState or kUpdateState
 */
TVM_DLL CallEffectKind SideEffect(const PrimExpr& expr);

/*!
 * \brief Whether the given Stmt uses any var in the given variable set.
 * \param stmt The Stmt to be checked.
 * \param vset_contains The check function to see if a var is in the variable set.
 * \return Whether `stmt` uses any var in the given variable set.
 */
TVM_DLL bool UsesVar(const Stmt& stmt, std::function<bool(const VarNode*)> vset_contains);

/*!
 * \brief Whether the given PrimExpr uses any var in the given variable set.
 * \param expr The PrimExpr to be checked.
 * \param vset_contains The check function to see if var is in the variable set.
 * \return Whether `expr` uses any var in the given variable set.
 */
TVM_DLL bool UsesVar(const PrimExpr& expr, std::function<bool(const VarNode*)> vset_contains);

/*!
 * \brief Whether the given Stmt reads any var in the given variable set.
 * \param stmt The Stmt to be checked.
 * \param vset_contains The check function to see if a var is in the variable set.
 * \return Whether `stmt` uses any var in the given variable set.
 */
TVM_DLL bool ReadsVar(const Stmt& stmt, std::function<bool(const VarNode*)> vset_contains);

/*!
 * \brief Whether the given Stmt writes any var in the given variable set.
 * \param stmt The Stmt to be checked.
 * \param vset_contains The check function to see if a var is in the variable set.
 * \return Whether `stmt` uses any var in the given variable set.
 */
TVM_DLL bool WritesVar(const Stmt& stmt, std::function<bool(const VarNode*)> vset_contains);

/*!
 * \brief Whether the given condition preserve the same value under the given scope.
 * \param condition The condition expression to be checked.
 * \param body The stmt scope to be checked.
 * \return Whether `stmt` uses any var in the given variable set.
 */
TVM_DLL bool IsPureCondition(const PrimExpr& condition, const Stmt& body);

/*!
 * \brief Verifies whether the IR stmt or Expr is in SSA form.
 *  That is: each Var is defined and assigned once(in Let/For)
 *
 * \param func The function to be verified.
 * \return Whether IR is in SSA form.
 *
 * \note All passes in TIR consume and produce SSA form.
 */
TVM_DLL bool VerifySSA(const PrimFunc& func);

/*!
 * \brief Verify if memory accesses are legal for a specific target device type.
 *
 *  In the case that tgt is cuda, if not all workload is bound with
 *  threads, CPU code is generated that tries to access GPU memory,
 *  which is illegal. This pass performs verification for this case.
 *
 * \param func The function to be verified.
 * \return Success of memory verification.
 */
TVM_DLL bool VerifyMemory(const PrimFunc& func);

/*!
 * \brief Calculate the expression complexity based on number of symbols it contains.
 * \param expr The expr to be calculated.
 */
TVM_DLL size_t CalculateExprComplexity(const PrimExpr& expr);

/*!
 * \brief Calculate the constants size in bytes needed by the TIR allocates inside the TIR PrimFunc
 * \param func The TIR PrimFunc for which the constants size to be calculated
 * \param constant_byte_alignment The byte alignment required for each constant allocated
 */
TVM_DLL size_t CalculateConstantBytes(const PrimFunc& func, const Integer& constant_byte_alignment);

/*!
 * \brief Calculate the workspace size in bytes needed by the TIR allocates inside the TIR PrimFunc
 * \param func The TIR PrimFunc for which the workspace size to be calculated
 * \param workspace_byte_alignment The byte alignment required for each tensor allocated in this
 * workspace
 */
TVM_DLL size_t CalculateWorkspaceBytes(const PrimFunc& func,
                                       const Integer& workspace_byte_alignment);

/*!
 * \brief Verify if the given TIR is well-formed. The verification includes:
 *
 * - All variables are defined prior to their point of use.
 *
 * - No variables are used outside of the scope of their definition.
 *
 * - Each variable has a single point of definition.
 *
 * - Expressions within a tir::SBlock may not reference variables
 *   defined outside the block.  For example, for a block with iter
 *   vars `vi, vj = T.axis.remap('SS', [i,j])`, the statement
 *   `B[i,j] = A[i,j]` would be ill-formed, because it uses the loop
 *   variables `i` and `j` instead of the block variables `vi` and
 *   `vj`.
 *
 * \param func The PrimFunc to be verified.
 * \param assert_mode The indicator if it raises an error when the function is not well-formed.
 * \return Whether it is a well-formed TIR function.
 */
TVM_DLL bool VerifyWellFormed(const PrimFunc& func, bool assert_mode = true);

/*!
 * \brief Verify if the TIR in the given IRMOdule is well-formed.
 *
 * In addition to the checks performed for each PrimFunc (see above),
 * the following checks are performed:
 *
 * - The same TIR variable may not be defined in more than one function
 *
 * \param mod The IRModule to be verified.
 * \param assert_mode The indicator if it raises an error when the function is not well-formed.
 * \return Whether it is a well-formed TIR module.
 */
TVM_DLL bool VerifyWellFormed(const IRModule& mod, bool assert_mode = true);

/*!
 * \brief Find the entry function of the given IRModule, i.e, functions marked by
 * `tir::attr::kIsEntryFunc`, whose name is `main` or being the only PrimeFunc.
 * \param mod The IRModule to find the entry function.
 * \param result_g_var The result GlobalVar of the entry function.
 * \return The entry function.
 */
const PrimFuncNode* FindEntryFunc(const IRModule& mod, GlobalVar* result_g_var);

// Pass variants of verification analysis
// directly throws RuntimeError when verification fails.
namespace transform {

using tvm::transform::Pass;
using tvm::transform::PassContext;

/*!
 * \brief Pass variant of VerifySSA.
 *
 * \returns The pass.
 * \sa tvm::tir::VerifySSA
 */
TVM_DLL Pass VerifySSA();

/*!
 * \brief Pass variant of VerifyMemory.
 *
 * \returns The pass.
 * \sa tvm::tir::VerifyMemory
 */
TVM_DLL Pass VerifyMemory();

}  // namespace transform
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_ANALYSIS_H_
