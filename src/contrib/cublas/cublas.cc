/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external cblas library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>

extern "C" {
#include <cublas_v2.h>
}

namespace tvm {
namespace contrib {

using namespace runtime;

#ifndef CHECK_CUBLAS_ERROR
#define CHECK_CUBLAS_ERROR(error) \
if (error != CUBLAS_STATUS_SUCCESS) { \
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

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cublas.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* A = args[0];
    DLTensor* B = args[1];
    DLTensor* C = args[2];
    bool transa = args[3];
    bool transb = args[4];
    // call gemm for simple compact code.
    CHECK_EQ(A->ndim, 2);
    CHECK_EQ(B->ndim, 2);
    CHECK_EQ(C->ndim, 2);
    CHECK(C->strides == nullptr);
    CHECK(B->strides == nullptr);
    CHECK(A->strides == nullptr);
    CHECK(TypeMatch(A->dtype, kDLFloat, 32));
    CHECK(TypeMatch(B->dtype, kDLFloat, 32));
    CHECK(TypeMatch(C->dtype, kDLFloat, 32));

    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    float alpha = 1.0;
    float beta = 0.0;
    float *A_ptr = reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset);
    float *B_ptr = reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset);
    float *C_ptr = reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset);

    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                   transb ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   transb ? B->shape[0] : B->shape[1],
                                   transa ? A->shape[1] : A->shape[0],
                                   transb ? B->shape[1] : B->shape[0],
                                   &alpha,
                                   A_ptr,
                                   B->shape[1],
                                   B_ptr,
                                   A->shape[1],
                                   &beta,
                                   C_ptr,
                                   C->shape[1]));

    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
});
}  // namespace contrib
}  // namespace tvm
