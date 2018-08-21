/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/alpha_eq.h
 * \brief Check expressions & types for structural equivalence.
 */
#ifndef TVM_RELAY_ALPHA_EQ_H_
#define TVM_RELAY_ALPHA_EQ_H_

#include "tvm/relay/ir.h"

namespace tvm {
namespace relay {

bool alpha_eq(const Expr & e1, const Expr & e2);
bool alpha_eq(const Type & t1, const Type & t2);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ALPHA_EQ_H_
