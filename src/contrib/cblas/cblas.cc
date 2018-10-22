/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external cblas library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>

extern "C" {
#if USE_MKL_BLAS == 1
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
}

namespace tvm {
namespace contrib {

using namespace runtime;

  inline int column_stride(DLTensor* tensor)
  {
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

  inline CBLAS_TRANSPOSE boolean_to_transpose( bool trans ) {
    return trans ? CblasTrans : CblasNoTrans;
  }

  template<typename TGemmOp>
  inline void call_gemm(TVMArgs args, TVMRetValue *ret, int bit_depth, TGemmOp op) {
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];
    bool transa = args[3];
    bool transb = args[4];
    CHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
    CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));

    op(CblasColMajor,
       boolean_to_transpose(transb),
       boolean_to_transpose(transa),
       column_count(B, transb),
       column_count(A, transa),
       row_count(A, transa),
       1.0f,
       reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(B->data) + B->byte_offset),
       column_stride(B),
       reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(A->data) + A->byte_offset),
       column_stride(A),
       0.0f,
       reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(C->data) + C->byte_offset),
       column_stride(C));
  }

  struct sgemm_op
  {
    typedef float TDatatype;
    void operator()(CBLAS_ORDER order, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K,
		    float alpha, float* A, int lda,
		    float* B, int ldb,
		    float beta, float* C, int ldc) {
      cblas_sgemm(order, ta, tb, M, N, K,
		  alpha, A, lda,
		  B, ldb,
		  beta, C, ldc);
    }
  };

  struct dgemm_op
  {
    typedef double TDatatype;
    void operator()(CBLAS_ORDER order, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K,
		    double alpha, double* A, int lda,
		    double* B, int ldb,
		    double beta, double* C, int ldc) {
      cblas_dgemm(order, ta, tb, M, N, K,
		  alpha, A, lda,
		  B, ldb,
		  beta, C, ldc);
    }
  };


// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cblas.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];
    // call gemm for simple compact code.
    CHECK_EQ(A->ndim, 2);
    CHECK_EQ(B->ndim, 2);
    CHECK_EQ(C->ndim, 2);
    CHECK(TypeMatch(A->dtype, kDLFloat, 32) ||
	  TypeMatch(A->dtype, kDLFloat, 64));

    if (TypeMatch(A->dtype, kDLFloat, 32))
    {
      call_gemm(args, ret, 32, sgemm_op());
    }
    else if (TypeMatch(A->dtype, kDLFloat, 64))
    {
      call_gemm(args, ret, 64, dgemm_op());
    }
  });
}  // namespace contrib
}  // namespace tvm
