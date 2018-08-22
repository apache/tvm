/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/options.h
 * \brief Global options for the Relay IR.
 */
#ifndef TVM_RELAY_TYPECK_RESOLVE_H_
#define TVM_RELAY_TYPECK_RESOLVE_H_

#include <string>
#include "tvm/relay/ir.h"
#include "./unifier.h"

namespace tvm {
namespace relay {

Type resolve(const TypeUnifier & unifier, const Type & ty);
Expr resolve(const TypeUnifier & unifier, const Expr & expr);
bool is_fully_resolved(const Type & t);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TYPECK_RESOLVE_H_
