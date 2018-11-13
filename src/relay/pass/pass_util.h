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

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_PASS_UTIL_H_
