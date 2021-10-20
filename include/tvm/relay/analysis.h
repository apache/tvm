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
 * \file tvm/relay/analysis.h
 * \brief The set of Relay analysis passes written in C++.
 */
#ifndef TVM_RELAY_ANALYSIS_H_
#define TVM_RELAY_ANALYSIS_H_

#include <tvm/ir/module.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/logging.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace relay {

/*!
 * \brief Check that types are well kinded by applying "kinding rules".
 *
 * This pass ensures we do not do things that violate the design of the
 * type system when writing down types.
 *
 * For example tensors are not allowed to contain functions in Relay.
 *
 * We check this by ensuring the `dtype` field of a Tensor always contains
 * a data type such as `int`, `float`, `uint`.
 *
 * \param t The type to check.
 * \param mod The global module.
 * \param diag_ctx The Diagnostic context.
 *
 * \return The kind of the passed type.
 */
TVM_DLL Kind KindCheck(const Type& t, const IRModule& mod,
                       Optional<DiagnosticContext> diag_ctx = Optional<DiagnosticContext>());

/*!
 * \brief Check whether an expression is constant.
 *
 * If the inputs of an expression are all constant, it means the expression
 * itself is constant also.
 *
 * \param e the expression.
 *
 * \return whether the expression is constant.
 */
TVM_DLL bool ConstantCheck(const Expr& e);

/*!
 * \brief Check whether an expression is in the basic block normal form.
 *
 * \param e the expression.
 *
 * \return whether the expression is in the basic block normal form.
 */
TVM_DLL bool BasicBlockNormalFormCheck(const Expr& e);

/*!
 * \brief Check that each Var is only bound once.
 *
 * For example, the expression `let x = 1 in let x = 2 in 3` bound x twice.
 *
 * `let f = (x -> x) in let g = (x -> x + 1) in f(g(2))` also bound x twice,
 * although x is not shadowed.
 *
 * \param expr the expression to check.
 * \param diag_ctx the diagnostic context
 *
 * \return true iff all Var in expr is bound at most once.
 */
TVM_DLL bool WellFormed(const Expr& expr,
                        Optional<DiagnosticContext> diag_ctx = Optional<DiagnosticContext>());

/*!
 * \brief Get all bound variables from expression expr.
 *
 * Bound variables are all variables that are declared in the expr.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param expr the expression.
 *
 * \return List of bound vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> BoundVars(const Expr& expr);

/*!
 * \brief Get all bound variables from pattern pat.
 *
 * Bound variables are all variables that got bound by the pat.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param pat the Pattern.
 *
 * \return List of bound vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> BoundVars(const Pattern& pat);

/*!
 * \brief Get free type parameters from expression expr.
 *
 * Free variables are variables that are not bound by a
 * let or a function parameter in the context.
 *
 * \param expr the expression.
 *
 * \return List of free vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> FreeVars(const Expr& expr);

/*!
 * \brief Get all variables from expression expr.
 *
 * \param expr the expression.
 *
 * \return List of all vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> AllVars(const Expr& expr);

/*!
 * \brief Get free TypeVars from expression expr.
 *
 * Free type parameters are type parameters that are not bound by a function
 * type in the context.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of free vars, in the PostDFS order visited by expr.
 */
TVM_DLL tvm::Array<TypeVar> FreeTypeVars(const Expr& expr, const IRModule& mod);

/*!
 * \brief Get free TypeVars from type t.
 *
 * Free type parameters are type parameters that are not bound by a function
 * type in the context.
 *
 * \param t the type.
 * \param mod the module.
 *
 * \return List of free type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> FreeTypeVars(const Type& t, const IRModule& mod);

/*!
 * \brief Get all bound type variables from expression expr.
 *
 * Bound variables are all type variables that are declared in the expr.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of bound type vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<TypeVar> BoundTypeVars(const Expr& expr, const IRModule& mod);

/*!
 * \brief Get all bound type variables from type t.
 *
 * Bound variables are all type variables that are declared in the type.
 * They only have meaning inside that type, and can only be used in it.
 *
 * \param t the type
 * \param mod the module.
 *
 * \return List of bound type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> BoundTypeVars(const Type& t, const IRModule& mod);

/*!
 * \brief Get all type variables in expression expr.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Expr& expr, const IRModule& mod);

/*!
 * \brief Get all type variables in type t.
 *
 * \param t the type.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Type& t, const IRModule& mod);

/*!
 * \brief Finds cases that the given match expression does not catch, if any.
 *
 * \param match the match expression to test
 *
 * \param mod The module used for accessing global type var definitions, can be None.
 *
 * \return Returns a list of cases (as patterns) that are not handled by the match
 * expression.
 */
TVM_DLL Array<Pattern> UnmatchedCases(const Match& match, const IRModule& mod);

/*!
 * \brief Get reference counter of each internal ExprNode in body.
 *
 * \param body The body expression.
 *
 * \return The reference count mapping.
 */
TVM_DLL std::unordered_map<const Object*, size_t> GetExprRefCount(const Expr& body);

/*!
 * \brief Get the updated module for collecting calibration data.
 *
 * \param mod The module to be updated.
 *
 * \return The updated module.
 */
TVM_DLL IRModule GetCalibrateModule(IRModule mod);

/*!
 * \brief Get the output map between subgrpahs and its inputs/output.
 *
 * \param mod The module for running calibration.
 *
 * \return The mapping between a subgraph name and its postition in the output tuple.
 */
TVM_DLL Map<GlobalVar, Array<Integer>> GetCalibrateOutputMap(const IRModule& mod);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ANALYSIS_H_
