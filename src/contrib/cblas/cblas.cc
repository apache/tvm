/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external cblas library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <tvm/contrib/gemm.h>


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

  inline CBLAS_TRANSPOSE boolean_to_transpose( bool trans ) {
    return trans ? CblasTrans : CblasNoTrans;
  }

  struct sgemm_op
  {
    typedef float TDatatype;
    void operator()(bool ta, bool tb,
		    int M, int N, int K,
		    float alpha, float* A, int lda,
		    float* B, int ldb,
		    float beta, float* C, int ldc) {
      cblas_sgemm(CblasColMajor,
		  boolean_to_transpose(ta),
		  boolean_to_transpose(tb),
		  M, N, K,
		  alpha, A, lda,
		  B, ldb,
		  beta, C, ldc);
    }
  };

  struct dgemm_op
  {
    typedef double TDatatype;
    void operator()(bool ta, bool tb,
		    int M, int N, int K,
		    double alpha, double* A, int lda,
		    double* B, int ldb,
		    double beta, double* C, int ldc) {
      cblas_dgemm(CblasColMajor,
		  boolean_to_transpose(ta),
		  boolean_to_transpose(tb),
		  M, N, K,
		  alpha, A, lda,
		  B, ldb,
		  beta, C, ldc);
    }
  };


// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cblas.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* A = args[0];
    CHECK(TypeMatch(A->dtype, kDLFloat, 32) ||
	  TypeMatch(A->dtype, kDLFloat, 64));

    if (TypeMatch(A->dtype, kDLFloat, 32))
      call_gemm(args, ret, sgemm_op());
    else
      call_gemm(args, ret, dgemm_op());
  });
}  // namespace contrib
}  // namespace tvm
