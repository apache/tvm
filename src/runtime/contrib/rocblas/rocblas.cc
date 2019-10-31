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
 *  Copyright (c) 2017 by Contributors
 * \file Use external rocblas library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include "rocblas.h"

namespace tvm {
namespace contrib {

using namespace runtime;

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(error) \
if (error != rocblas_status_success) { \
  fprintf(stderr, "rocBLAS error: "); \
  if (error == rocblas_status_invalid_handle) fprintf(stderr, "rocblas_status_invalid_handle"); \
  if (error == rocblas_status_not_implemented) fprintf(stderr, " rocblas_status_not_implemented"); \
  if (error == rocblas_status_invalid_pointer) fprintf(stderr, "rocblas_status_invalid_pointer"); \
  if (error == rocblas_status_invalid_size) fprintf(stderr, "rocblas_status_invalid_size"); \
  if (error == rocblas_status_memory_error) fprintf(stderr, "rocblas_status_memory_error"); \
  if (error == rocblas_status_internal_error) fprintf(stderr, "rocblas_status_internal_error"); \
  fprintf(stderr, "\n"); \
  exit(EXIT_FAILURE); \
}
#endif


// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.rocblas.matmul")
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

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));
    float alpha = 1.0;
    float beta = 0.0;
    float *A_ptr = reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset);
    float *B_ptr = reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset);
    float *C_ptr = reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset);

    CHECK_ROCBLAS_ERROR(rocblas_sgemm(handle,
                                      transb ? rocblas_operation_transpose : rocblas_operation_none,
                                      transa ? rocblas_operation_transpose : rocblas_operation_none,
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

    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
});
}  // namespace contrib
}  // namespace tvm
