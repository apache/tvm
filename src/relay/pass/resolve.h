/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/resolve.h
 * \brief Resolve incomplete types to complete types.
 */
#ifndef TVM_RELAY_PASS_RESOLVE_H_
#define TVM_RELAY_PASS_RESOLVE_H_

#include <tvm/relay/expr.h>
#include <string>
#include "./unifier.h"

namespace tvm {
namespace relay {


/*! \brief Resolve a type containing incomplete types.
*
* This pass replaces incomplete types with their representative, and 
* converts types which are not defined into fresh variables.
* 
* \param unifier The unifier containing the unification data.
* \param ty The type to resolve.
* \returns The resolved type.
*/
Type Resolve(const TypeUnifier & unifier, const Type & ty);

/*! \brief Resolve an expression containing incomplete types.
*
* This pass replaces incomplete types with their representative, and 
* converts types which are not defined into fresh variables.
* 
* \param unifier The unifier containing the unification data.
* \param ty The expression to resolve.
* \returns The resolved expression.
*/
Expr Resolve(const TypeUnifier & unifier, const Expr & expr);

/*! \brief Check if all types have been filled in. 
*   \param t The type.
*   \returns True if the type is resolved, false otherwise.
*/
bool IsFullyResolved(const Type & t);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_RESOLVE_H_
