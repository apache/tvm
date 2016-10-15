/*!
 *  Copyright (c) 2016 by Contributors
 * \file tensor.h
 * \brief Dataflow tensor object
 */
#ifndef TVM_TENSOR_H_
#define TVM_TENSOR_H_

#include "./expr.h"

namespace tvm {

class Tensor {
 private:
  /*! \brief The shape of the tensor */
  /*! \brief source expression */
  Expr src_expr;
};


}  // namespace tvm
#endif  // TVM_TENSOR_H_
