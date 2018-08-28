/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/pass.h
 * \brief The set of Relay passes written in C++.
 */
#ifndef TVM_RELAY_PASS_H_
#define TVM_RELAY_PASS_H_

#include "tvm/relay/expr.h"
#include "tvm/relay/environment.h"

namespace tvm {
namespace relay {

/*! The result of type checking an expression is a new expression
 * with unambigous type information filled in, as well as it's
 * checked type field populated with the result type.
 */
Expr InferType(const Environment & env, const Expr & e);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPECHECKER_H_