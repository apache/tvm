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
 * \file tvm/tir/ir_pass.h
 * \brief Collection of IR pass functions
 *
 *  When the pass functions in this file are for Stmt,
 *  we can use PassFunction(Evaluate(expr)) to apply it to Expr
 */
#ifndef TVM_TIR_IR_PASS_H_
#define TVM_TIR_IR_PASS_H_

#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/function.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>


namespace tvm {
namespace tir {

/*!
 * \brief Simplify the expression.
 * \param expr The expression to be simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized statement.
 */
TVM_DLL PrimExpr Simplify(PrimExpr expr, Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief Simplify the statement.
 * \param stmt The statement to be simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized statement.
 */
Stmt Simplify(Stmt stmt, Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief Simplify by applying canonical form.
 * \param stmt The statement to be canonically simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized statement.
 */
Stmt CanonicalSimplify(Stmt stmt,
                       Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief Simplify by applying canonical form.
 * \param expr The statement to be canonically simplifed.
 * \param vrange The range information about the variable.
 * \return Canonicalized expression.
 */
TVM_DLL PrimExpr CanonicalSimplify(PrimExpr expr,
                                   Map<Var, Range> vrange = Map<Var, Range>());

/*!
 * \brief verifies whether the IR stmt or Expr is in SSA form.
 *  That is: each VarExpr is defined and assigned once(in Let/For)
 *
 * \param ir The root of the IR DAG.
 * \return Whether IR is in SSA form.
 * \note All the passes in this file uses SSA form and outputs SSA form.
 */
TVM_DLL bool VerifySSA(const Stmt& ir);

/*!
 * \brief Whether the expression have side effect.
 * \return whether expression have side effect
 */
TVM_DLL bool HasSideEffect(const PrimExpr& e);

/*!
 * \brief Whether e expression used var.
 * \param e The expression to be checked.
 * \param v The variable.
 * \return Whether e uses v.
 */
bool ExprUseVar(const PrimExpr& e, const Var& v);

/*!
 * \brief Whether e expression used any var in variable set..
 * \param e The expression to be checked.
 * \param vset The variable set.
 * \return Whether e uses vset.
 */
bool ExprUseVar(const PrimExpr& e, const std::unordered_set<const VarNode*>& vset);

/*!
 * \brief Convert a IR node to be SSA form.
 * \param stmt The source statement to be converted.
 * \return The converted form.
 */
TVM_DLL Stmt ConvertSSA(Stmt stmt);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param stmt The source statement to be substituted
 * \param value_map The map of new values.
 * \return The converted form.
 */
Stmt Substitute(Stmt stmt,
                const std::unordered_map<const VarNode*, PrimExpr>& value_map);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param expr The source expression to be substituted
 * \param value_map The map of new values.
 * \return The converted expression.
 */
PrimExpr Substitute(PrimExpr expr,
                const std::unordered_map<const VarNode*, PrimExpr>& value_map);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param stmt The source statement to be substituted
 * \param value_map The map of new values.
 * \return The converted form.
 */
Stmt Substitute(Stmt stmt, const Map<Var, PrimExpr>& value_map);

/*!
 * \brief Substitute the var specified in key->var to be value.
 * \param expr The source expression to be substituted
 * \param value_map The map of new values.
 * \return The converted expression.
 */
PrimExpr Substitute(PrimExpr expr, const Map<Var, PrimExpr>& value_map);

/*!
 * \brief inline all calls of f in stmt.
 *
 * \param stmt The statement to apply inline optimization.
 * \param f The function reference to be inlined
 * \param args The arguments variable of the function.
 * \param body The definition body of the function.
 * \return The result stmt
 *
 * \note All the passes in this file uses SSA form and outputs SSA form.
 */
Stmt Inline(Stmt stmt,
            FunctionRef f,
            Array<Var> args,
            PrimExpr body);

/*!
 * \brief Verify if there is any argument bound to compact buffer.
 *
 * \param stmt The stmt to be verified.
 * \return true if there is any buffer_bind_scope attribute found,
 *        otherwise, false.
 */
bool VerifyCompactBuffer(Stmt stmt);

/*!
 * \brief Decorate the stmt with a device scope, this is helpful for
 * hardware accelerator without thread blocks.
 *
 * \param stmt The stmt to be transformed
 * \return Transformed stmt.
 */
Stmt DecorateDeviceScope(Stmt stmt);

/*!
 * \brief Loop invariant code motion which locates and hoists if statements.
 * \param stmt The stmt to do if statement hoisting.
 * \return Transformed stmt.
 */
Stmt HoistIfThenElse(Stmt stmt);

/*!
 * \brief Rewrite the pointer content type of arguments,
 *  as well as Alloc internal to the function to use
 *  the most frequently accessed type for load/store
 *  to avoid pointer casting in backend when possible.
 *
 * \note implemeneted in storage_rewrite.cc
 * \param f The function to be trasnformed
 * \return Transformed function.
 */
PrimFunc PointerValueTypeRewrite(PrimFunc f);

/*!
 * \brief Verify the correctness of a GPU code
 *        It will check the whether the amount of memory usage or the number of threads
 *        in a block exceeds the limit
 * \param stmt The statement to be checked
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
bool VerifyGPUCode(Stmt stmt,
                   Map<std::string, PrimExpr> constraints);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_PASS_H_
