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
 * \file tvm/relay/pass.h
 * \brief The set of Relay passes written in C++.
  */
#ifndef TVM_RELAY_PASS_H_
#define TVM_RELAY_PASS_H_

#include <tvm/ir.h>
#include <tvm/packed_func_ext.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/type.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

/*!
 * \brief Infer the type of an expression.
 *
 * The result of type checking is a new expression with unambigous
 * type information filled in, as well as it's checked type field
 * populated with the result type.
 *
 * \param expr The expression to type check.
 * \param mod The module used for referencing global functions, can be
 * None.
 *
 * \return A type checked expression with its checked_type field populated.
 */
TVM_DLL Expr InferType(const Expr& expr, const Module& mod);

/*!
 * \brief Infer the type of a function as if it is mapped to var in the mod.
 *
 * \param f the function.
 * \param mod The module used for referencing global functions.
 * \param var The global variable corresponding to the function.
 *
 * \return A type checked Function with its checked_type field populated.
 * \note this function mutates mod and is not thread-safe.
 */
TVM_DLL Function InferType(const Function& f, const Module& mod,
                           const GlobalVar& var);

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
 * \brief Fold constant expressions.
 *
 * \param expr the expression to be optimized.
 *
 * \return The optimized expression.
 */
TVM_DLL Expr FoldConstant(const Expr& expr);

/*!
 * \brief Fuse operations into expr into seperate functions.
 *
 * \param expr The expression.
 * \param fuse_opt_level Optimization level.
 * \param mod the module.
 *
 * \return The optimized expression.
 */
TVM_DLL Expr FuseOps(const Expr& expr, int fuse_opt_level, const Module& mod);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order.
 *
 * \param expr The expression.
 * \param rewrite_map_attr_name The Op's attr name which corresponds to the rewrite
 *                              rule function.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                            const std::string& rewrite_map_attr_name,
                            std::function<NodeRef(const Call&)> fcontext = nullptr,
                            std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order.
 *
 * \param expr The expression.
 * \param rewrite_func The rewrite func that will apply to all operators.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 *
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                            const FForwardRewrite& rewrite_func,
                            std::function<NodeRef(const Call&)> fcontext = nullptr,
                            std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Rewrite the annotated program.
 *
 * \param expr The expression.
 * \param fallback_device The fallback device which is the default device for
 *                        operators without annotation.
 *
 * \return The updated program.
 */
TVM_DLL Expr RewriteAnnotatedOps(const Expr& expr, int fallback_device);

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

/*!
 * \brief Bind the free variables to a Relay expression.
 *
 * Parameter binding can only happen if expr is a Function.
 * binds cannot change internal arguments of internal functions.
 *
 * \param expr The function to be binded.
 * \param binds The map of arguments to
 *
 * \return The expression with all free vars bound.
 */
TVM_DLL Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds);

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

namespace vm {

/*!
 * \brief Compile a module, and construct the virtual machine.
 *
 * \param mod The module to compile.
 *
 * \return The constructed virtual machine.
 */
runtime::vm::VirtualMachine CompileModule(const Module& mod);

}  // namespace vm

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_H_
