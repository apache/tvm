/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Use external rocblas library call.
 */
#include "rocblas.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace contrib {

using namespace runtime;

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(error)                                                                \
  if (error != rocblas_status_success) {                                                          \
    fprintf(stderr, "rocBLAS error: ");                                                           \
    if (error == rocblas_status_invalid_handle) fprintf(stderr, "rocblas_status_invalid_handle"); \
    if (error == rocblas_status_not_implemented)                                                  \
      fprintf(stderr, " rocblas_status_not_implemented");                                         \
    if (error == rocblas_status_invalid_pointer)                                                  \
      fprintf(stderr, "rocblas_status_invalid_pointer");                                          \
    if (error == rocblas_status_invalid_size) fprintf(stderr, "rocblas_status_invalid_size");     \
    if (error == rocblas_status_memory_error) fprintf(stderr, "rocblas_status_memory_error");     \
    if (error == rocblas_status_internal_error) fprintf(stderr, "rocblas_status_internal_error"); \
    fprintf(stderr, "\n");                                                                        \
    exit(EXIT_FAILURE);                                                                           \
  }
#endif

struct RocBlasThreadEntry {
  RocBlasThreadEntry() { CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle)); }

  ~RocBlasThreadEntry() {
    if (handle) {
      CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
      handle = nullptr;
    }
  }

  rocblas_handle handle;
};  // RocBlasThreadEntry

typedef dmlc::ThreadLocalStore<RocBlasThreadEntry> RocBlasThreadStore;

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.rocblas.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  // call gemm for simple compact code.
  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);
  ICHECK(C->strides == nullptr);
  ICHECK(B->strides == nullptr);
  ICHECK(A->strides == nullptr);
  ICHECK(TypeMatch(A->dtype, kDLFloat, 32));
  ICHECK(TypeMatch(B->dtype, kDLFloat, 32));
  ICHECK(TypeMatch(C->dtype, kDLFloat, 32));

  float alpha = 1.0;
  float beta = 0.0;
  float* A_ptr = reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset);
  float* B_ptr = reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset);
  float* C_ptr = reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset);

  rocblas_operation roc_trans_A = transa ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation roc_trans_B = transb ? rocblas_operation_transpose : rocblas_operation_none;
  size_t N = transb ? B->shape[0] : B->shape[1];
  size_t M = transa ? A->shape[1] : A->shape[0];
  size_t K = transb ? B->shape[1] : B->shape[0];
  size_t lda = transa ? M : K;
  size_t ldb = transb ? K : N;
  size_t ldc = N;

  CHECK_ROCBLAS_ERROR(rocblas_sgemm(RocBlasThreadStore::Get()->handle, roc_trans_B, roc_trans_A, N,
                                    M, K, &alpha, B_ptr, ldb, A_ptr, lda, &beta, C_ptr, ldc));
});

TVM_REGISTER_GLOBAL("tvm.contrib.rocblas.batch_matmul")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* A = args[0];
      DLTensor* B = args[1];
      DLTensor* C = args[2];
      bool transa = args[3];
      bool transb = args[4];
      // call gemm for simple compact code.
      ICHECK_EQ(A->ndim, 3);
      ICHECK_EQ(B->ndim, 3);
      ICHECK_EQ(C->ndim, 3);
      ICHECK(TypeMatch(A->dtype, kDLFloat, 32));
      ICHECK(TypeMatch(B->dtype, kDLFloat, 32));
      ICHECK(TypeMatch(C->dtype, kDLFloat, 32));

      float alpha = 1.0;
      float beta = 0.0;
      float* A_ptr = reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset);
      float* B_ptr = reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset);
      float* C_ptr = reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset);

      rocblas_operation roc_trans_A = transa ? rocblas_operation_transpose : rocblas_operation_none;
      rocblas_operation roc_trans_B = transb ? rocblas_operation_transpose : rocblas_operation_none;
      size_t batch_size = C->shape[0];
      size_t N = transb ? B->shape[1] : B->shape[2];
      size_t M = transa ? A->shape[2] : A->shape[1];
      size_t K = transb ? B->shape[2] : B->shape[1];
      size_t lda = transa ? M : K;
      size_t ldb = transb ? K : N;
      size_t ldc = N;

      CHECK_ROCBLAS_ERROR(rocblas_sgemm_strided_batched(
          RocBlasThreadStore::Get()->handle, roc_trans_B, roc_trans_A, N, M, K, &alpha, B_ptr, ldb,
          K * N, A_ptr, lda, M * K, &beta, C_ptr, ldc, M * N, batch_size));
    });
}  // namespace contrib
}  // namespace tvm
