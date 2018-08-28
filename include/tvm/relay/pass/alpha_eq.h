/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/alpha_eq.h
 * \brief Check expressions and types for structural equivalence.
 */
#ifndef TVM_RELAY_ALPHA_EQ_H_
#define TVM_RELAY_ALPHA_EQ_H_

#include "../type.h"
#include "../expr.h"

namespace tvm {
namespace relay {

bool AlphaEqual(const Expr & e1, const Expr & e2);
bool AlphaEqual(const Type & t1, const Type & t2);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ALPHA_EQ_H_
