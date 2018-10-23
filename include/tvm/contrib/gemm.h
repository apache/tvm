/*!
 *  Copyright (c) 2016 by Contributors
 * \file tvm/contrib/gemm.h
 * \brief Shared implementation of gemm
 */
#ifndef TVM_CONTRIB_GEMM_H_
#define TVM_CONTRIB_GEMM_H_
#include <tvm/runtime/registry.h>

namespace tvm {
namespace contrib {

  using namespace runtime;

  inline int column_stride(DLTensor* tensor) {
    if (tensor->strides)
      return tensor->strides[0];
    else
      return tensor->shape[1];
  }

  inline int row_count(DLTensor* tensor, bool trans) {
    return tensor->shape[trans ? 1 : 0];
  }

  inline int column_count(DLTensor* tensor, bool trans) {
    return tensor->shape[trans ? 0 : 1];
  }

  // Call a column major blas.  Note that data is stored in tvm as row
  // major, so this we switch the arguments.
  template<typename TGemmOp>
  inline void call_gemm(TVMArgs args, TVMRetValue *ret, TGemmOp op) {
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];
    bool transa = args[3];
    bool transb = args[4];
    int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
    CHECK_EQ(A->ndim, 2);
    CHECK_EQ(B->ndim, 2);
    CHECK_EQ(C->ndim, 2);
    CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
    CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
    double alpha = args.size() > 5 ? args[5] : 1.0;
    double beta = args.size() > 6 ? args[6] : 0.0;
    op(transb,
       transa,
       column_count(B, transb),
       column_count(A, transa),
       row_count(A, transa),
       static_cast<float>(alpha),
       reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(B->data)
                                                      + B->byte_offset),
       column_stride(B),
       reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(A->data)
                                                      + A->byte_offset),
       column_stride(A),
       static_cast<float>(beta),
       reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(C->data)
                                                      + C->byte_offset),
       column_stride(C));
  }

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_GEMM_H_
