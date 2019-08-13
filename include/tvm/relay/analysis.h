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

#include <tvm/relay/adt.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/type.h>
#include <string>

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
 *
 * \return The kind of the passed type.
 */
TVM_DLL Kind KindCheck(const Type& t, const Module& mod);

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
 * \brief Compare two expressions for structural equivalence.
 *
 * This comparison operator respects scoping and compares
 * expressions without regard to variable choice.
 *
 * For example: `let x = 1 in x` is equal to `let y = 1 in y`.
 *
 *   See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
 *   for more details.
 *
 *   \param e1 The left hand expression.
 *   \param e2 The right hand expression.
 *
 *   \return true if equal, otherwise false
 */
TVM_DLL bool AlphaEqual(const Expr& e1, const Expr& e2);

/*!
 * \brief Compare two types for structural equivalence.
 *
 * This comparison operator respects scoping and compares
 * expressions without regard to variable choice.
 *
 * For example: `forall s, Tensor[f32, s]` is equal to
 * `forall w, Tensor[f32, w]`.
 *
 * See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
 * for more details.
 *
 * \param t1 The left hand type.
 * \param t2 The right hand type.
 *
 * \return true if equal, otherwise false
 */
TVM_DLL bool AlphaEqual(const Type& t1, const Type& t2);

/*!
 * \brief Compare two patterns for structural equivalence.
 *
 * This comparison operator respects scoping and compares
 * patterns without regard to variable choice.
 *
 * For example: `A(x, _, y)` is equal to `A(z, _, a)`.
 *
 * See https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence
 * for more details.
 *
 * \param t1 The left hand pattern.
 * \param t2 The right hand pattern.
 *
 * \return true if equal, otherwise false
 */
TVM_DLL bool AlphaEqual(const Pattern& t1, const Pattern& t2);

/*!
 * \brief Check that each Var is only bound once.
 *
 * For example, the expression `let x = 1 in let x = 2 in 3` bound x twice.
 *
 * `let f = (\x -> x) in let g = (\x -> x + 1) in f(g(2))` also bound x twice,
 * although x is not shadowed.
 *
  * \param expr the expression to check.
 *
  * \return true iff all Var in expr is bound at most once.
 */
TVM_DLL bool WellFormed(const Expr& expr);

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
TVM_DLL tvm::Array<TypeVar> FreeTypeVars(const Expr& expr, const Module& mod);

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
TVM_DLL tvm::Array<TypeVar> FreeTypeVars(const Type& t, const Module& mod);

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
TVM_DLL tvm::Array<TypeVar> BoundTypeVars(const Expr& expr, const Module& mod);

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
TVM_DLL tvm::Array<TypeVar> BoundTypeVars(const Type& t, const Module& mod);

/*!
 * \brief Get all type variables in expression expr.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Expr& expr, const Module& mod);

/*!
 * \brief Get all type variables in type t.
 *
 * \param t the type.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Type& t, const Module& mod);

/*!
 * \brief Collect the device mapping information of each expression.
 *
 * \param expr The expression.
 *
 * \return The device mapping.
 */
TVM_DLL Map<Expr, Integer> CollectDeviceInfo(const Expr& expr);

/*!
 * \brief Collect the device anntation operators.
 *
 * \param expr The expression.
 *
 * \return The annotated expression to device type mapping for annotation ops.
 */
TVM_DLL Map<Expr, Integer> CollectDeviceAnnotationOps(const Expr& expr);

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
TVM_DLL Array<Pattern> UnmatchedCases(const Match& match, const Module& mod);

/*! \brief A hashing structure in the style of std::hash. */
struct StructuralHash {
  /*! \brief Hash a Relay type.
   *
   * Implements structural hashing of a Relay type.
   *
   * \param type the type to hash.
   *
   * \return the hash value.
   */
  size_t operator()(const Type& type) const;
  /*! \brief Hash a Relay expression.
   *
   * Implements structural hashing of a Relay expression.
   *
   * \param expr the expression to hash.
   *
   * \return the hash value.
   */
  size_t operator()(const Expr& expr) const;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ANALYSIS_H_
