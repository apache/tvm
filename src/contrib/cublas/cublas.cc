/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external cblas library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <tvm/contrib/blas.h>

extern "C" {
#include <cublas_v2.h>
}

namespace tvm {
namespace contrib {

using namespace runtime;

#ifndef CHECK_CUBLAS_ERROR
#define CHECK_CUBLAS_ERROR(_fn_) \
if (int error = _fn_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error: "); \
  if (error == CUBLAS_STATUS_NOT_INITIALIZED) fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED"); \
  if (error == CUBLAS_STATUS_ALLOC_FAILED) fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED"); \
  if (error == CUBLAS_STATUS_INVALID_VALUE) fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE"); \
  if (error == CUBLAS_STATUS_ARCH_MISMATCH) fprintf(stderr, "CUBLAS_STATUS_ARCH_MISMATCH"); \
  if (error == CUBLAS_STATUS_MAPPING_ERROR) fprintf(stderr, "CUBLAS_STATUS_MAPPING_ERROR"); \
  if (error == CUBLAS_STATUS_EXECUTION_FAILED) fprintf(stderr, "CUBLAS_STATUS_EXECUTION_FAILED"); \
  if (error == CUBLAS_STATUS_INTERNAL_ERROR) fprintf(stderr, "CUBLAS_STATUS_INTERNAL_ERROR"); \
  if (error == CUBLAS_STATUS_NOT_SUPPORTED) fprintf(stderr, "CUBLAS_STATUS_NOT_SUPPORTED"); \
  if (error == CUBLAS_STATUS_LICENSE_ERROR) fprintf(stderr, "CUBLAS_STATUS_LICENSE_ERROR"); \
  fprintf(stderr, "\n"); \
  exit(EXIT_FAILURE); \
}
#endif

  inline int boolean_to_transpose(bool item) {
    return item ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  struct sgemm_op
  {
    typedef float TDatatype;
    cublasHandle_t handle;
    sgemm_op( cublasHandle_t hdl )
      : handle(hdl)
      {}

    void operator()(bool ta, bool tb,
		    int M, int N, int K,
		    float alpha, float* A, int lda,
		    float* B, int ldb,
		    float beta, float* C, int ldc) {
      CHECK_CUBLAS_ERROR(cublasSgemm(handle,
				     boolean_to_transpose(ta),
				     boolean_to_transpose(tb),
				     M, N, K,
				     alpha, A, lda,
				     B, ldb,
				     beta, C, ldc));
  };


  struct dgemm_op
  {
    typedef double TDatatype;
    cublasHandle_t handle;
    dgemm_op( cublasHandle_t hdl )
      : handle(hdl)
      {}
    void operator()(bool ta, bool tb,
		    int M, int N, int K,
		    double alpha, double* A, int lda,
		    double* B, int ldb,
		    double beta, double* C, int ldc) {
      CHECK_CUBLAS_ERROR(cublasDgemm(handle,
				     boolean_to_transpose(ta),
				     boolean_to_transpose(tb),
				     M, N, K,
				     alpha, A, lda,
				     B, ldb,
				     beta, C, ldc));
    }
  };

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cublas.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {

    DLTensor* A = args[0];


    CHECK(TypeMatch(A->dtype, kDLFloat, 32) ||
	  TypeMatch(A->dtype, kDLFloat, 64));

    TVMStreamHandle stream = TVMGetStream(A->device_type, A->device_id);

    DeviceAPIManager::Get(ctx)->SetStream(ctx, stream);

    //TODO cache handle
    //TODO set stream appropriate to the thread current stream.
    static cublasHandle_t handle = 0;
    if (handle == 0) {
      CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    }

    if (stream)
      cublasSetStream(static_cast<cudaStream_t>(stream));

    if (TypeMatch(A->dtype, kDLFloat, 32))
      call_gemm(args, ret, sgemm_op(handle));
    else
      call_gemm(args, ret, dgemm_op(handle));
});
}  // namespace contrib
}  // namespace tvm
