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
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;

    cublasSgemm(handle,
                transb ? CUBLAS_OP_T : CUBLAS_OP_N,
                transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                transb ? B->shape[0] : B->shape[1],
                transa ? A->shape[1] : A->shape[0],
                transb ? B->shape[1] : B->shape[0],
                &alpha,
                reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset),
                B->shape[1],
                reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset),
                A->shape[1],
                &beta,
                reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset),
                C->shape[1]);

    cublasDestroy(handle);
});
}  // namespace contrib
}  // namespace tvm
