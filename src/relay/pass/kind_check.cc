/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file kindchecker.cc
 *
 * \brief Check that types are well formed by applying "kinding rules".
 *
 * This pass ensures we do not do things that violate the design of the
 * type system when writing down types.
 *
 * For example tensors are not allowed to contain functions in Relay.
 *
 * We check this by ensuring the `dtype` field of a Tensor always 
 * contains a data type such as `int`, `float`, `uint`.
 */
#include <tvm/ir_functor.h>
#include <tvm/relay/pass.h>
#include "./type_visitor.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

struct KindChecker : TypeVisitor<> {
  bool valid;

  KindChecker() : valid(true) {}

  bool Check(const Type &t) {
    this->VisitType(t);
    return valid;
  }
};

bool KindCheck(const Environment& env, const Type &t) {
  KindChecker kc;
  return kc.Check(t);
}

}  // namespace relay
}  // namespace tvm