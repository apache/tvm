/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/relay/typechecker.h
 * \brief Type check a Relay program producing a type checked program
 *  with its checked_type field populated and incomplete types resolved.
 */
#ifndef TVM_RELAY_COMPILER_TYPECHECKER_H_
#define TVM_RELAY_COMPILER_TYPECHECKER_H_

#include "tvm/relay/ir.h"
#include "tvm/relay/environment.h"

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
