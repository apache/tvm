/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/pass.h
 * \brief The set of Relay passes written in C++.
 */
#ifndef TVM_RELAY_PASS_H_
#define TVM_RELAY_PASS_H_

#include <tvm/relay/environment.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

/*! \brief Infer the type of an expression with the provided environment.
 *
 * The result of type checking is a new expression with unambigous
 * type information filled in, as well as it's checked type field
 * populated with the result type.
 *
 * \param env The environment used for global settings and referencing
 * global functions.
 *
 * \param e The expression to type check.
 *
 * \return A type checked expression with its checked_type field populated.
 */
Expr InferType(const Environment& env, const Expr& e);
Expr InferType(const Environment& env, const GlobalVar& v, const Function& e);

/*!
 * \brief Check that types are well formed by applying "kinding rules".
 *
 * This pass ensures we do not do things that violate the design of the
 * type system when writing down types.
 *
 * For example tensors are not allowed to contain functions in Relay.
 *
 * We check this by ensuring the `dtype` field of a Tensor always contains
 * a data type such as `int`, `float`, `uint`.
 *
 * \param env The global environment.
 * \param t The type to check.
 * \return true if the rules are satisified otherwise false
 */
bool KindCheck(const Environment& env, const Type& t);

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
bool AlphaEqual(const Expr& e1, const Expr& e2);

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
bool AlphaEqual(const Type& t1, const Type& t2);

/*! brief Check that no LocalVar got shadowed.
 *
 * Roughly speaking, a LocalVar is considered to be shadowed, if it was introduce while it was already in scoped.
 *
 * For example, the expression `let x = 1 in let x = 2 in 3` shadow x.
 *
 * However, `let f = (\x -> x) in let g = (\x -> x + 1) in f(g(2))` does not shadow x, f, g.
 * x is not shadowed because it is introduce at other scoped - the x used by f is invisible to the x used by g.
 *
 * \param e the expression to check.
 *
 * \return true iff e has no shadowing.
 */
 bool LocalVarWellFormed(const Expr & e);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_H_
