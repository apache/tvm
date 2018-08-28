/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/resolve.h
 * \brief Resolve incomplete types to complete types.
 */
#ifndef TVM_RELAY_TYPECK_RESOLVE_H_
#define TVM_RELAY_TYPECK_RESOLVE_H_

#include <string>
#include <tvm/relay/expr.h>
#include "./unifier.h"

namespace tvm {
namespace relay {

Type Resolve(const TypeUnifier & unifier, const Type & ty);
Expr Resolve(const TypeUnifier & unifier, const Expr & expr);
bool IsFullyResolved(const Type & t);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TYPECK_RESOLVE_H_
