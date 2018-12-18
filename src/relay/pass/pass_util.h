/*!
 *  Copyright (c) 2018 by Contributors.
 *
 * \file tvm/relay/pass/pass_util.h
 * \brief Utilities for writing
 */
#ifndef TVM_RELAY_PASS_PASS_UTIL_H_
#define TVM_RELAY_PASS_PASS_UTIL_H_

#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/attrs/transform.h>

namespace tvm {
namespace relay {

/*!
 * \brief Get reference counter of each internal ExprNode in body.
 * \param body The body expression.
 * \return The reference count mapping.
 */
std::unordered_map<const Node*, size_t>
GetExprRefCount(const Expr& body);


/*!
 * \brief Check if expr is positive constant.
 * \param expr The expression to be checked.
 * \return Whether all elements of expr is positive constant.
 */
bool IsAllPositiveConstant(const Expr& expr);


/*!
 * \brief Substitute var with subst.
 * \param type The type to be substituted.
 * \param tvar The type variable to be substituted.
 * \param subst The target of substitution.
 * \return The substituted result.
 */
Type TypeSubst(const Type& type, const TypeVar& tvar, const Type& subst);

/*!
 * \brief Substitute type vars in type.
 * \param type The type to be substituted.
 * \param subst_map The map of substitution.
 * \return The substituted result.
 */
Type TypeSubst(const Type& type, const tvm::Map<TypeVar, Type>& subst_map);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_PASS_UTIL_H_
