/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/contrib/gemm.h
 * \brief Shared implementation of gemm
 */
#ifndef TVM_CONTRIB_GEMM_DETAILS_H_
#define TVM_CONTRIB_GEMM_DETAILS_H_
#include <algorithm>
#include "dlpack/dlpack.h"


namespace tvm {
namespace contrib {

using namespace runtime;


inline int columnStride(DLTensor* tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return std::max(tensor->strides[0], tensor->strides[1]);
  } else {
    return tensor->shape[1];
  }
}


inline int elementStride(DLTensor* tensor) {
  if (tensor->strides) {
    return std::min(tensor->strides[0], tensor->strides[1]);
  } else {
    return 1;
  }
}


// Reversed strides indicates an in-place transpose operation.
inline bool isInPlaceTransposed(DLTensor* tensor) {
  return tensor->strides && (tensor->strides[1] > tensor->strides[0]);
}


inline int rowCount(DLTensor* tensor, bool trans) {
  return tensor->shape[trans ? 1 : 0];
}


inline int columnCount(DLTensor* tensor, bool trans) {
  return tensor->shape[trans ? 0 : 1];
}


}  // namespace contrib
}  // namespace tvm


#endif  // TVM_CONTRIB_GEMM_DETAILS_H_
