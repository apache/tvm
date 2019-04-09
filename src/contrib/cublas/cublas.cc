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
 *  Copyright (c) 2018 by Contributors
 * \file Use external cblas library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include "../cblas/gemm_common.h"
#include "cublas_utils.h"


namespace tvm {
namespace contrib {

using namespace runtime;

inline cublasOperation_t BooleanToTranspose(bool item) {
  return item ? CUBLAS_OP_T : CUBLAS_OP_N;
}

struct CublasSgemmOp {
  typedef float TDatatype;
  cublasHandle_t handle;
  explicit CublasSgemmOp(cublasHandle_t hdl)
    : handle(hdl)
    {}

  void operator()(bool ta, bool tb,
                  int M, int N, int K,
                  float alpha, float* A, int lda,
                  float* B, int ldb,
                  float beta, float* C, int ldc) {
    CHECK_CUBLAS_ERROR(cublasSgemm(handle,
                                   BooleanToTranspose(ta),
                                   BooleanToTranspose(tb),
                                   M, N, K,
                                   &alpha, A, lda,
                                   B, ldb,
                                   &beta, C, ldc));
  }
};


struct CublasDgemmOp {
  typedef double TDatatype;
  cublasHandle_t handle;
  explicit CublasDgemmOp(cublasHandle_t hdl)
    : handle(hdl)
    {}
  void operator()(bool ta, bool tb,
                  int M, int N, int K,
                  double alpha, double* A, int lda,
                  double* B, int ldb,
                  double beta, double* C, int ldc) {
    CHECK_CUBLAS_ERROR(cublasDgemm(handle,
                                   BooleanToTranspose(ta),
                                   BooleanToTranspose(tb),
                                   M, N, K,
                                   &alpha, A, lda,
                                   B, ldb,
                                   &beta, C, ldc));
  }
};

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cublas.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* A = args[0];

    CHECK(TypeMatch(A->dtype, kDLFloat, 32) ||
          TypeMatch(A->dtype, kDLFloat, 64));

    CuBlasThreadEntry* entry_ptr = CuBlasThreadEntry::ThreadLocal();

    if (TypeMatch(A->dtype, kDLFloat, 32))
      CallGemm(args, ret, CublasSgemmOp(entry_ptr->handle));
    else
      CallGemm(args, ret, CublasDgemmOp(entry_ptr->handle));
});
}  // namespace contrib
}  // namespace tvm
