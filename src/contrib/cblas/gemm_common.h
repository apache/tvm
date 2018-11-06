/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/contrib/gemm.h
 * \brief Shared implementation of gemm
 */
#ifndef TVM_CONTRIB_CBLAS_GEMM_COMMON_H_
#define TVM_CONTRIB_CBLAS_GEMM_COMMON_H_
#include <algorithm>

namespace tvm {
namespace contrib {

using namespace runtime;

inline int ColumnStride(DLTensor* tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return std::max(tensor->strides[0], tensor->strides[1]);
  } else {
    return tensor->shape[1];
  }
}


inline int ElementStride(DLTensor* tensor) {
  if (tensor->strides) {
    return std::min(tensor->strides[0], tensor->strides[1]);
  } else {
    return 1;
  }
}


// Reversed strides indicates an in-place transpose operation.
inline bool IsInPlaceTransposed(DLTensor* tensor) {
  return tensor->strides && (tensor->strides[1] > tensor->strides[0]);
}


inline int RowCount(DLTensor* tensor, bool trans) {
  return tensor->shape[trans ? 1 : 0];
}


inline int ColumnCount(DLTensor* tensor, bool trans) {
  return tensor->shape[trans ? 0 : 1];
}

// Call a column major blas.  Note that data is stored in tvm as row
// major, so this we switch the arguments.
template<typename TGemmOp>
inline void CallGemm(TVMArgs args, TVMRetValue *ret, TGemmOp op) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  CHECK_EQ(ElementStride(A), 1);
  CHECK_EQ(ElementStride(B), 1);
  CHECK_EQ(ElementStride(C), 1);

  // C can never be transposed.
  CHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;
  op(transb,
     transa,
     ColumnCount(B, transb),
     RowCount(A, transa),
     ColumnCount(A, transa),
     static_cast<float>(alpha),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(B->data)
                                                    + B->byte_offset),
     ColumnStride(B),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(A->data)
                                                    + A->byte_offset),
     ColumnStride(A),
     static_cast<float>(beta),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(C->data)
                                                    + C->byte_offset),
     ColumnStride(C));
}

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_CBLAS_GEMM_COMMON_H_
