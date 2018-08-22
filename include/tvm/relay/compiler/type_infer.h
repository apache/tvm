/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/type_infer.h
 * \brief Perform type inference and checking on Relay programs.
 *
 *  The pass produces a new expression with its checked_type
 *  field populated and incomplete types resolved.
 */
#ifndef TVM_RELAY_COMPILER_TYPECHECKER_H_
#define TVM_RELAY_COMPILER_TYPECHECKER_H_

#include "tvm/relay/ir.h"
#include "tvm/relay/compiler/environment.h"

namespace tvm {
namespace relay {

/*! The result of type checking an expression is a new expression
 * with unambigous type information filled in, as well as it's
 * checked type field populated with the result type.
 */
Expr check(const Environment & env, const Expr & e);
Operator check(const Environment & env, const Operator & op);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_COMPILER_TYPECHECKER_H_
