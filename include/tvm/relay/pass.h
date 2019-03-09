/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/pass.h
 * \brief The set of Relay passes written in C++.
 */
#ifndef TVM_RELAY_PASS_H_
#define TVM_RELAY_PASS_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/op_attr_types.h>
#include <string>

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

/*! \brief Compare two expressions for structural equivalence.
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

/*! \brief Compare two types for structural equivalence.
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

/*! \brief Check that each Var is only bound once.
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

/*! \brief Get all bound variables from expression expr.
 *
 * Bound variables are all variables that are declared in the expr.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param expr the expression.
 *
 * \return List of bound vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> BoundVars(const Expr& expr);

/*! \brief Get free type parameters from expression expr.
 *
 * Free variables are variables that are not bound by a
 * let or a function parameter in the context.
 *
 * \param expr the expression.
 *
 * \return List of free vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> FreeVars(const Expr& expr);

/*! \brief Get all variables from expression expr.
 *
 * \param expr the expression.
 *
 * \return List of all vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<Var> AllVars(const Expr& expr);

/*! \brief Get free TypeVars from expression expr.
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

/*! \brief Get free TypeVars from type t.
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

/*! \brief Get all bound type variables from expression expr.
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

/*! \brief Get all bound type variables from type t.
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

/*! \brief Get all type variables in expression expr.
 *
 * \param expr the expression.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Expr& expr, const Module& mod);

/*! \brief Get all type variables in type t.
 *
 * \param t the type.
 * \param mod the module.
 *
 * \return List of type vars, in the PostDFS order visited by type.
 */
TVM_DLL tvm::Array<TypeVar> AllTypeVars(const Type& t, const Module& mod);

/*! \brief Remove expressions which does not effect the program result.
 *
 * It will remove let bindings which are not referenced, and branches that will
 * not be entered.
 *
 * For example, this pass should turn `let a = 1 in 2` into `2`, as the value of
 * the expression does not depend on a. Another example is `if (true) then 1
 * else 2` will be optimized into 1.
 *
 * \param e the expression to optimize.
 *
 * \return the optimized expression.
 */
TVM_DLL Expr DeadCodeElimination(const Expr& e);

/*!
 * \brief Fold constant expressions.
 * \param expr the expression to be optimized.
 * \return The optimized expression.
 */
TVM_DLL Expr FoldConstant(const Expr& expr);

/*!
 * \brief Fuse operations into expr into seperate functions.
 * \param expr The expression.
 * \param fuse_opt_level Optimization level.
 * \return The optimized expression.
 */
TVM_DLL Expr FuseOps(const Expr& expr, int fuse_opt_level);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order.
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
 * \param expr The expression.
 * \param rewrite_func The rewrite func that will apply to all operators.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                    const FForwardRewrite& rewrite_func,
                    std::function<NodeRef(const Call&)> fcontext = nullptr,
                    std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Rewrite the annotated program.
 * \param expr The expression.
 * \param fallback_device The fallback device which is the default device for
 *                        operators without annotation.
 * \return The updated program.
 */
TVM_DLL Expr RewriteAnnotatedOps(const Expr& expr, int fallback_device);

/*!
 * \brief Collect the device mapping information of each expression.
 * \param expr The expression.
 * \return The device mapping.
 */
TVM_DLL Map<Expr, Integer> CollectDeviceInfo(const Expr& expr);

/*! \brief A hashing structure in the style of std::hash. */
struct StructuralHash {
  /*! \brief Hash a Relay type.
   *
   * Implements structural hashing of a Relay type.
   *
   *  \param type the type to hash.
   *
   *  \return the hash value.
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

/*! \brief turn a dataflow graph into Administrative Normal Form, or A-Normal Form (ANF).
 *
 * It will turn an expression that is in a graph form (with sharing implicit),
 * to an expression with explicit sharing (A-Normal Form).
 *
 * The scope of the root expression is the global scope.

 * The scope of any non root expression is the least common ancestor of all it's scope.
 *
 * Values are ordered by post-DFS order in each scope.
 *
 * \param e the expression to observably share
 *
 * \param mod The module used for referencing global functions, can be
 * None.
 *
 * \return expression in A-Normal Form
 */
Expr ToANormalForm(const Expr& e, const Module& mod);

/*! \brief Remove let binding and directly share via pointer instead.
 *
 * It will remove all let binding,
 * and turn all of the variable bound by let into direct pointer reference.
 *
 * \param e the expression.
 *
 * \return the expression in graph normal form.
 */
Expr ToGraphNormalForm(const Expr& e);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_H_
