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

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPECHECKER_H_