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
 * \file Use external cuSPARSE library call.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include "cusparse_utils.h"


namespace tvm {
namespace contrib {

using namespace runtime;

inline cusparseOperation_t BooleanToTranspose(bool item) {
  return item ? CUSPARSE_OPERATION_TRANSPOSE : 
                CUSPARSE_OPERATION_NON_TRANSPOSE;
}


struct CuSparseScsrmmOp {
  typedef float TDatatype;
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  explicit CuSparseScsrmmOp(cusparseHandle_t hdl, cusparseMatDescr_t des)
    : handle(hdl), descr(des)
    {}

  void operator()(bool ta, int M, int N, int K, int NNZ,
                  float alpha, float* valA, int* rowPtrA, int* colIndA,
                  float* B, int ldb, float beta,
                  float* C, int ldc) {
    CHECK_CUSPARSE_ERROR(cusparseScsrmm(handle,
                                        BooleanToTranspose(ta),
                                        M, N, K, NNZ,
                                        &alpha, descr, 
                                        valA, rowPtrA, colIndA,
                                        B, ldb, &beta,
                                        C, ldc));
  }
};


struct CuSparseDcsrmmOp {
  typedef double TDatatype;
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  explicit CuSparseDcsrmmOp(cusparseHandle_t hdl, cusparseMatDescr_t des)
    : handle(hdl), descr(des)
    {}
  void operator()(bool ta, int M, int N, int K, int NNZ,
                  double alpha, double* valA, int* rowPtrA, int* colIndA,
                  double* B, int ldb, double beta,
                  double* C, int ldc) {
    CHECK_CUSPARSE_ERROR(cusparseDcsrmm(handle,
                                        BooleanToTranspose(ta),
                                        M, N, K, NNZ,
                                        &alpha, descr, 
                                        valA, rowPtrA, colIndA,
                                        B, ldb, &beta,
                                        C, ldc));
  }
};


// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.cusparse.matmul")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    DLTensor* A = args[0];
    CHECK(TypeMatch(A->dtype, kDLFloat, 32) ||
          TypeMatch(A->dtype, kDLFloat, 64));

    CuSparseThreadEntry* entry_ptr = CuSparseThreadEntry::ThreadLocal();

    if (TypeMatch(A->dtype, kDLFloat, 32))
      CallCsrmm(args, ret, CuSparseScsrmmOp(entry_ptr->handle, entry_ptr->descr));
    else
      CallCsrmm(args, ret, CuSparseDcsrmmOp(entry_ptr->handle, entry_ptr->descr));
});

}  // namespace contrib
}  // namespace tvm
