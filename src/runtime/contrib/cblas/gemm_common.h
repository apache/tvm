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
 * \file tvm/contrib/gemm.h
 * \brief Shared implementation of gemm
 */

#ifndef TVM_RUNTIME_CONTRIB_CBLAS_GEMM_COMMON_H_
#define TVM_RUNTIME_CONTRIB_CBLAS_GEMM_COMMON_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace contrib {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::TypeMatch;

inline int ColumnStride(const DLTensor* tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return std::max(tensor->strides[0], tensor->strides[1]);
  } else {
    return tensor->shape[1];
  }
}

inline int ElementStride(const DLTensor* tensor) {
  if (tensor->strides) {
    return std::min(tensor->strides[0], tensor->strides[1]);
  } else {
    return 1;
  }
}

// Reversed strides indicates an in-place transpose operation.
inline bool IsInPlaceTransposed(const DLTensor* tensor) {
  return tensor->strides && (tensor->strides[1] > tensor->strides[0]);
}

inline int RowCount(const DLTensor* tensor, bool trans, int batch_offset = 0) {
  return tensor->shape[batch_offset + (trans ? 1 : 0)];
}

inline int ColumnCount(const DLTensor* tensor, bool trans, int batch_offset = 0) {
  return tensor->shape[batch_offset + (trans ? 0 : 1)];
}

// Call a column major blas.  Note that data is stored in tvm as row
// major, so this we switch the arguments.
template <typename TGemmOp>
inline void CallGemm(TVMArgs args, TVMRetValue* ret, TGemmOp op) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];
  int bit_depth = sizeof(typename TGemmOp::TDatatype) * 8;
  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);

  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(C), 1);

  // C can never be transposed.
  ICHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  ICHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  ICHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;
  op(transb, transa, ColumnCount(B, transb), RowCount(A, transa), ColumnCount(A, transa),
     static_cast<typename TGemmOp::TDatatype>(alpha),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(B->data) + B->byte_offset),
     ColumnStride(B),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(A->data) + A->byte_offset),
     ColumnStride(A), static_cast<typename TGemmOp::TDatatype>(beta),
     reinterpret_cast<typename TGemmOp::TDatatype*>(static_cast<char*>(C->data) + C->byte_offset),
     ColumnStride(C));
}

// Call a column major blas.  Note that data is stored in tvm as row
// major, so this we switch the arguments.
template <typename TGemmOp>
inline void CallU8S8S32Gemm(TVMArgs args, TVMRetValue* ret, TGemmOp op) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];

  // Set the sgemm attributes. Currently, support is limited to CblasFixOffset with all offsets
  // equal to 0. This is sufficient for relay dense.
  std::string offset_ctype = "CblasFixOffset";
  int16_t offset_a = 0;
  int16_t offset_b = 0;
  int offset_c[1];
  offset_c[0] = 0;

  ICHECK_EQ(A->ndim, 2);
  ICHECK_EQ(B->ndim, 2);
  ICHECK_EQ(C->ndim, 2);

  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(C), 1);

  // C can never be transposed.
  ICHECK(!IsInPlaceTransposed(C));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  ICHECK(TypeMatch(A->dtype, kDLUInt, 8));
  ICHECK(TypeMatch(B->dtype, kDLInt, 8));
  ICHECK(TypeMatch(C->dtype, kDLInt, 32));
  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;
  op(transb, transa, ColumnCount(B, transb), RowCount(A, transa), ColumnCount(A, transa),
     static_cast<float>(alpha),
     reinterpret_cast<void*>(static_cast<char*>(B->data) + B->byte_offset), ColumnStride(B),
     offset_b, reinterpret_cast<void*>(static_cast<char*>(A->data) + A->byte_offset),
     ColumnStride(A), offset_a, static_cast<float>(beta),
     reinterpret_cast<int*>(static_cast<char*>(C->data) + C->byte_offset), ColumnStride(C),
     offset_ctype, offset_c);
}

inline int ColumnStride3D(DLTensor* tensor) {
  // If the tensor itself is transposed then it will have strides
  // backward from what we expect.  Regardless, the max of the strides
  // (the other stride is 1) is the column stride.
  if (tensor->strides) {
    return std::max(tensor->strides[1], tensor->strides[2]);
  } else {
    return tensor->shape[2];
  }
}
inline int ElementStride3D(const DLTensor* tensor) {
  if (tensor->strides) {
    return std::min(tensor->strides[1], tensor->strides[2]);
  } else {
    return 1;
  }
}
// Reversed strides indicates an in-place transpose operation.
inline bool IsInPlaceTransposed3D(DLTensor* tensor) {
  return tensor->strides && (tensor->strides[2] > tensor->strides[1]);
}
inline int BatchCount3D(DLTensor* tensor) { return tensor->shape[0]; }
inline int RowCount3D(DLTensor* tensor, bool trans) { return tensor->shape[trans ? 2 : 1]; }
inline int ColumnCount3D(DLTensor* tensor, bool trans) { return tensor->shape[trans ? 1 : 2]; }
template <typename TBatchGemmOp>
inline void CallBatchGemm(TVMArgs args, TVMRetValue* ret, TBatchGemmOp op) {
  using DType = typename TBatchGemmOp::TDatatype;
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  bool transa = args[3];
  bool transb = args[4];

  int bit_depth = sizeof(DType) * 8;

  ICHECK_EQ(A->ndim, 3);
  ICHECK_EQ(B->ndim, 3);
  ICHECK_EQ(C->ndim, 3);

  int batch_size = BatchCount3D(C);
  ICHECK_EQ(ElementStride(A), 1);
  ICHECK_EQ(ElementStride(B), 1);
  ICHECK_EQ(ElementStride(C), 1);

  // C can never be transposed.
  ICHECK(!IsInPlaceTransposed3D(C));
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed3D(A) ? !transa : transa;
  transb = IsInPlaceTransposed3D(B) ? !transb : transb;

  ICHECK(TypeMatch(B->dtype, kDLFloat, bit_depth));
  ICHECK(TypeMatch(C->dtype, kDLFloat, bit_depth));

  double alpha = args.size() > 5 ? args[5] : 1.0;
  double beta = args.size() > 6 ? args[6] : 0.0;

  int A_stride = A->shape[1] * A->shape[2];
  int B_stride = B->shape[1] * B->shape[2];
  int C_stride = C->shape[1] * C->shape[2];

  // Broadcast A or B by changing its stride.
  int batch_size_a = BatchCount3D(A);
  int batch_size_b = BatchCount3D(B);
  if (batch_size_a != batch_size_b) {
    if (batch_size_a == 1) {
      A_stride = 0;
    } else if (batch_size_b == 1) {
      B_stride = 0;
    }
  } else {
    ICHECK_EQ(batch_size_a, batch_size);
    ICHECK_EQ(batch_size_b, batch_size);
  }

  DType* A_data = reinterpret_cast<typename TBatchGemmOp::TDatatype*>(static_cast<char*>(A->data) +
                                                                      A->byte_offset);
  DType* B_data = reinterpret_cast<typename TBatchGemmOp::TDatatype*>(static_cast<char*>(B->data) +
                                                                      B->byte_offset);
  DType* C_data = reinterpret_cast<typename TBatchGemmOp::TDatatype*>(static_cast<char*>(C->data) +
                                                                      C->byte_offset);
  op(batch_size, transb, transa, ColumnCount3D(B, transb), RowCount3D(A, transa),
     ColumnCount3D(A, transa), static_cast<typename TBatchGemmOp::TDatatype>(alpha), B_data,
     B_stride, ColumnStride3D(B), A_data, A_stride, ColumnStride3D(A),
     static_cast<typename TBatchGemmOp::TDatatype>(beta), C_data, C_stride, ColumnStride3D(C));
}

}  // namespace contrib
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_CBLAS_GEMM_COMMON_H_
