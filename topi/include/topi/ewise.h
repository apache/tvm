/*!
 *  Copyright (c) 2017 by Contributors
 * \file topi.h
 * \brief Elementwise op constructions
 */
#ifndef TOPI_EWISE_H_
#define TOPI_EWISE_H_

#include <tvm/tvm.h>

namespace topi {
using namespace tvm;

// Unary intrinsic operators
#define TOPI_DECLARE_UNARY_OP(OpName)                                   \
  inline Tensor OpName(const Tensor& x) {                               \
    return compute(x->shape, [&](const Array<Var>& i) {                 \
        return ::tvm::OpName(x(i));                                     \
      });                                                               \
  }

TOPI_DECLARE_UNARY_OP(exp);
TOPI_DECLARE_UNARY_OP(tanh);
TOPI_DECLARE_UNARY_OP(sigmoid);
}  // namespace topi
#endif  // TOPI_EWISE_H_
