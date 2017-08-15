/*!
 *  Copyright (c) 2017 by Contributors
 * \file elemwise.h
 * \brief Elementwise op constructions
 */
#ifndef TOPI_ELEMWISE_H_
#define TOPI_ELEMWISE_H_

#include <string>

#include "topi/tags.h"
#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

// Unary intrinsic operators
#define TOPI_DECLARE_UNARY_OP(OpName)                           \
  inline Tensor OpName(const Tensor& x,                         \
                       std::string name = "tensor",             \
                       std::string tag = kElementWise) {        \
    return compute(x->shape, [&](const Array<Var>& i) {         \
        return ::tvm::OpName(x(i));                             \
      }, name, tag);                                            \
  }

TOPI_DECLARE_UNARY_OP(exp);
TOPI_DECLARE_UNARY_OP(tanh);
TOPI_DECLARE_UNARY_OP(sigmoid);
TOPI_DECLARE_UNARY_OP(sqrt);

}  // namespace topi
#endif  // TOPI_ELEMWISE_H_
