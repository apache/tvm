/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/pass/type_infer.h
 * \brief Perform type inference and checking on Relay programs.
 *
 *  The pass produces a new expression with its checked_type
 *  field populated and incomplete types resolved.
 */
#ifndef TVM_RELAY_PASS_TYPECHECKER_H_
#define TVM_RELAY_PASS_TYPECHECKER_H_

#include "tvm/relay/expr.h"
#include "tvm/relay/environment.h"

namespace tvm {
namespace relay {

/*! \brief Ensures that an operator is well-formed with respect
 * to Relay's type system.
 */
Op CheckOp(const Environment & env, const Op & op);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPECHECKER_H_
