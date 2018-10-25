/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/contrib/gemm.h
 * \brief Shared implementation of gemm
 */
#ifndef TVM_CONTRIB_GEMM_H_
#define TVM_CONTRIB_GEMM_H_
#include <algorithm>
#include "tvm/contrib/gemm_details.h"

namespace tvm {
namespace contrib {

using namespace runtime;

// Call a column major blas.  Note that data is stored in tvm as row
// major, so this we switch the arguments.
template<typename TGemmOp>
inline void callGemm(TVMArgs args, TVMRetValue *ret, TGemmOp op) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(B->ndim, 2);
  CHECK_EQ(C->ndim, 2);

  CHECK_EQ(elementStride(A), 1);
  CHECK_EQ(elementStride(B), 1);
  CHECK_EQ(elementStride(C), 1);

  // C can never be transposed.
  CHECK(!isInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = isInPlaceTransposed(A) ? !transa : transa;
  transb = isInPlaceTransposed(B) ? !transb : transb;

  CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;
  op(transb,
     transa,
     columnCount(B, transb),
     rowCount(A, transa),
     columnCount(A, transa),
     static_cast<float>(alpha),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(B->data)
                                                    + B->byte_offset),
     columnStride(B),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(A->data)
                                                    + A->byte_offset),
     columnStride(A),
     static_cast<float>(beta),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(C->data)
                                                    + C->byte_offset),
     columnStride(C));
}

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_GEMM_H_
