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
 * \brief Analysis utilitie and passes for TIR.
 */
#ifndef TVM_TIR_ANALYSIS_H_
#define TVM_TIR_ANALYSIS_H_

#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <string>

namespace tvm {
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
 * \brief Find undefined vars in the statment.
 * \param stmt The function to be checked.
 * \param defs The vars that is defined.
 * \return Array of undefined vars.
 */
TVM_DLL Array<Var> UndefinedVars(const Stmt& stmt, const Array<Var>& defs);

/*!
 * \brief Whether the expression have side effect.
 * \param expr The expression to be checked.
 * \return whether expression have side effect
 */
TVM_DLL bool HasSideEffect(const PrimExpr& expr);

/*!
 * \brief Whether e expression used any var in variable set..
 * \param expr The expression to be checked.
 * \param vset_contains The check function to see if var is in the vset.
 * \return Whether e uses vset.
 */
TVM_DLL bool ExprUseVar(const PrimExpr& expr,
                        std::function<bool(const VarNode*)> vset_contains);

/*!
 * \brief Whether e expression used var.
 * \param expr The expression to be checked.
 * \param var The variable.
 * \return Whether e uses v.
 */
inline bool ExprUseVar(const PrimExpr& expr, const Var& var) {
  return ExprUseVar(expr, [&](const VarNode* node) {
    return var.get() == node;
  });
}


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
 * \brief Verify the correctness of a GPU code
 *        It will check the whether the amount of memory usage or the number of threads
 *        in a block exceeds the limit
 * \param func The function to be checked
 * \param constraints The dict to specify constraints to check.
 *        Possible keys are
 *
 *        "max_local_memory_per_block": Total amount of local memory per block (in bytes).
 *        "max_shared_memory_per_block": Total amount of shared memory per block (in bytes).
 *        "max_threads_per_block": Maximum number of threads per block.
 *        "max_thread_x": Maximum length of threadIdx.x.
 *        "max_thread_y": Maximum length of threadIdx.y.
 *        "max_thread_z": Maximum length of threadIdx.z.
 *
 *        If one key is missing in this argument, the pass won't check for that item.
 * \return valid Whether it is a valid GPU code
 *
 */
TVM_DLL bool VerifyGPUCode(const PrimFunc& func,
                           Map<std::string, PrimExpr> constraints);

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

/*!
 * \brief Pass variant of VerifyGPUCode.
 *
 * \param constraints The dict to specify constraints to check.
 *
 * \returns The pass.
 * \sa tvm::tir::VerifyGPUCode
 */
TVM_DLL Pass VerifyGPUCode(Map<std::string, PrimExpr> constraints);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_ANALYSIS_H_
