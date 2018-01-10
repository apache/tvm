/*!
*  Copyright (c) 2017 by Contributors
* \file ravel_unravel.h
* \brief Index ravel and unraval operations
*/
#ifndef TOPI_DETAIL_RAVEL_UNRAVEL_H_
#define TOPI_DETAIL_RAVEL_UNRAVEL_H_

#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

/*!
* \brief Flatten the indices to 1D
*
* \param indices The input coordinates
* \param shape Shape of the tensor
*
* \return The index after flattening
*/
inline Expr RavelIndex(Array<Var> indices, Array<Expr> shape) {
  CHECK_EQ(indices.size(), shape.size()) << "indices and shape must have equal size";
  CHECK_GT(indices.size(), 0) << "indices must not be empty";
  Expr idx;
  for (int i = 0; i < indices.size(); ++i) {
    if (i == 0) {
      idx = indices[i];
    } else {
      idx = idx * shape[i] + indices[i];
    }
  }
  return idx;
}

}  // namespace topi
#endif  // TOPI_DETAIL_RAVEL_UNRAVEL_H_
