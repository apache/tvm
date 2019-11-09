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
 * \file Use external cuSPARSE utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_CUSPARSE_CUSPARSE_UTILS_H_
#define TVM_RUNTIME_CONTRIB_CUSPARSE_CUSPARSE_UTILS_H_

#include <dmlc/logging.h>

#include <cusparse.h>
#include "../cblas/gemm_common.h"

namespace tvm {
namespace contrib {

inline const char* GetCuSparseErrorString(int error) {
  switch (error) {
  case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
  case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
  case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
  case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
  case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
  case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
  case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
  }
  return "Unrecognized error";
}

#ifndef CHECK_CUSPARSE_ERROR
#define CHECK_CUSPARSE_ERROR(fn)                  \
  do {                                          \
    int error = static_cast<int>(fn);                      \
    CHECK_EQ(error, CUSPARSE_STATUS_SUCCESS) << "CUSPARSE: " \
    << GetCuSparseErrorString(error); \
  } while (0)  // ; intentionally left off.
#endif  // CHECK_CUSPARSE_ERROR


struct CuSparseThreadEntry {
  CuSparseThreadEntry();
  ~CuSparseThreadEntry();
  cusparseHandle_t handle{nullptr};
  cusparseMatDescr_t descr{nullptr};
  static CuSparseThreadEntry* ThreadLocal();
};  // CuSparseThreadEntry

template <typename TCsrmmOp>
inline void CallCsrmm(TVMArgs args, TVMRetValue *ret, TCsrmmOp op) {
  DLTensor *A = args[0];
  DLTensor *valB = args[1];
  DLTensor *colIndB = args[2];
  DLTensor *rowPtrB = args[3];
  DLTensor *C = args[4];
  bool transb = args[5];

  int bit_depth = sizeof(typename TCsrmmOp::TDatatype) * 8;

  CHECK_EQ(A->ndim, 2);
  CHECK_EQ(valB->ndim, 1);
  CHECK_EQ(rowPtrB->ndim, 1);
  CHECK_EQ(colIndB->ndim, 1);
  CHECK_EQ(C->ndim, 2);

  CHECK_EQ(ElementStride(A), 1);
  CHECK_EQ(ElementStride(C), 1);

  // C can never be transposed.
  CHECK(!IsInPlaceTransposed(C));

  CHECK(TypeMatch(A->dtype, kDLFloat, bit_depth));
  CHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));

  double alpha = 1.0;
  double beta = 0.0;

  // only support square sparse matrix B for now
  op(transb, 
     // TODO pass in sp mat dim for flexible M, K
     static_cast<int>(rowPtrB->shape[0]-1), 
     static_cast<int>(A->shape[1]),
     // TODO pass in sp mat dim for flexible M, K
     static_cast<int>(rowPtrB->shape[0]-1),
     static_cast<int>(valB->shape[0]),
     static_cast<float>(alpha),
     reinterpret_cast<typename TCsrmmOp::TDatatype *>(
         static_cast<char *>(valB->data) + valB->byte_offset),
     reinterpret_cast<int *>(
         static_cast<char *>(rowPtrB->data) + rowPtrB->byte_offset),
     reinterpret_cast<int *>(
         static_cast<char *>(colIndB->data) + colIndB->byte_offset),
     reinterpret_cast<typename TCsrmmOp::TDatatype *>(
         static_cast<char *>(A->data) + A->byte_offset),
     ColumnStride(A),
     static_cast<float>(beta),
     reinterpret_cast<typename TCsrmmOp::TDatatype *>(
         static_cast<char *>(C->data) + C->byte_offset),
     ColumnStride(C));
}

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CUSPARSE_CUSPARSE_UTILS_H_
