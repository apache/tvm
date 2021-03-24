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
 * \file Use external mkl library call.
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

extern "C" {
#include <mkl_cblas.h>
}

#include "gemm_common.h"

namespace tvm {
namespace contrib {

using namespace runtime;
inline CBLAS_TRANSPOSE MKLBooleanToTranspose(bool trans) {
  return trans ? CblasTrans : CblasNoTrans;
}

inline CBLAS_OFFSET MKLStringToOffset(const std::string offset_type) {
  if (offset_type != "CblasFixOffset" && offset_type != "CblasColOffset" &&
      offset_type != "CblasRowOffset") {
    LOG(FATAL) << "Unrecognized offset_type " << offset_type;
  }
  if (offset_type == "CblasFixOffset") {
    return CblasFixOffset;
  } else if (offset_type == "CblasColOffset") {
    return CblasColOffset;
  }
  return CblasRowOffset;
}

inline char MKLBooleanToTransposeChar(bool trans) { return trans ? 'T' : 'N'; }

struct MKLGemmU8S8S32Op {
  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, const void* A, int lda,
                  int offset_a, const void* B, int ldb, int offset_b, float beta, int* C, int ldc,
                  const std::string offset_ctype, int* offset_c) {
    cblas_gemm_s8u8s32(CblasColMajor, MKLBooleanToTranspose(ta), MKLBooleanToTranspose(tb),
                       MKLStringToOffset(offset_ctype), M, N, K, alpha, A, lda, offset_a, B, ldb,
                       offset_b, beta, C, ldc, offset_c);
  }
};

struct MKLSgemmOp {
  typedef float TDatatype;
  void operator()(bool ta, bool tb, int M, int N, int K, float alpha, float* A, int lda, float* B,
                  int ldb, float beta, float* C, int ldc) {
    cblas_sgemm(CblasColMajor, MKLBooleanToTranspose(ta), MKLBooleanToTranspose(tb), M, N, K, alpha,
                A, lda, B, ldb, beta, C, ldc);
  }
};

struct MKLDgemmOp {
  typedef double TDatatype;
  void operator()(bool ta, bool tb, int M, int N, int K, double alpha, double* A, int lda,
                  double* B, int ldb, double beta, double* C, int ldc) {
    cblas_dgemm(CblasColMajor, MKLBooleanToTranspose(ta), MKLBooleanToTranspose(tb), M, N, K, alpha,
                A, lda, B, ldb, beta, C, ldc);
  }
};

struct MKLSgemmBatchOp {
  typedef float TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, float alpha, float* A,
                  int a_stride, int lda, float* B, int b_stride, int ldb, float beta, float* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = MKLBooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = MKLBooleanToTranspose(tb);
    std::vector<const float*> A_array(batch_size);
    std::vector<const float*> B_array(batch_size);
    std::vector<float*> C_array(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      A_array[i] = A + i * a_stride;
      B_array[i] = B + i * b_stride;
      C_array[i] = C + i * c_stride;
    }
    cblas_sgemm_batch(CblasColMajor, &trans_a, &trans_b, &M, &N, &K, &alpha, A_array.data(), &lda,
                      B_array.data(), &ldb, &beta, C_array.data(), &ldc, 1, &batch_size);
  }
};

struct MKLSgemmBatchIterativeOp {
  typedef float TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, float alpha, float* A,
                  int a_stride, int lda, float* B, int b_stride, int ldb, float beta, float* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = MKLBooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = MKLBooleanToTranspose(tb);
    for (int i = 0; i < batch_size; ++i) {
      cblas_sgemm(CblasColMajor, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      A += a_stride;
      B += b_stride;
      C += c_stride;
    }
  }
};

struct MKLDgemmBatchOp {
  typedef double TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, double alpha, double* A,
                  int a_stride, int lda, double* B, int b_stride, int ldb, double beta, double* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = MKLBooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = MKLBooleanToTranspose(tb);
    std::vector<const double*> A_array(batch_size);
    std::vector<const double*> B_array(batch_size);
    std::vector<double*> C_array(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      A_array[i] = A + i * a_stride;
      B_array[i] = B + i * b_stride;
      C_array[i] = C + i * c_stride;
    }
    cblas_dgemm_batch(CblasColMajor, &trans_a, &trans_b, &M, &N, &K, &alpha, A_array.data(), &lda,
                      B_array.data(), &ldb, &beta, C_array.data(), &ldc, 1, &batch_size);
  }
};

struct MKLDgemmBatchIterativeOp {
  typedef double TDatatype;
  void operator()(int batch_size, bool ta, bool tb, int M, int N, int K, double alpha, double* A,
                  int a_stride, int lda, double* B, int b_stride, int ldb, double beta, double* C,
                  int c_stride, int ldc) {
    CBLAS_TRANSPOSE trans_a = MKLBooleanToTranspose(ta);
    CBLAS_TRANSPOSE trans_b = MKLBooleanToTranspose(tb);
    for (int i = 0; i < batch_size; ++i) {
      cblas_dgemm(CblasColMajor, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      A += a_stride;
      B += b_stride;
      C += c_stride;
    }
  }
};

// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.mkl.matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  ICHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));

  if (TypeMatch(A->dtype, kDLFloat, 32))
    CallGemm(args, ret, MKLSgemmOp());
  else
    CallGemm(args, ret, MKLDgemmOp());
});

// integer matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.mkl.matmul_u8s8s32").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  DLTensor* B = args[1];
  DLTensor* C = args[2];
  ICHECK(TypeMatch(A->dtype, kDLUInt, 8) && TypeMatch(B->dtype, kDLInt, 8) &&
         TypeMatch(C->dtype, kDLInt, 32));

  CallU8S8S32Gemm(args, ret, MKLGemmU8S8S32Op());
});

TVM_REGISTER_GLOBAL("tvm.contrib.mkl.batch_matmul").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* A = args[0];
  ICHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));
  if (TypeMatch(A->dtype, kDLFloat, 32)) {
    CallBatchGemm(args, ret, MKLSgemmBatchOp());
  } else {
    CallBatchGemm(args, ret, MKLDgemmBatchOp());
  }
});

TVM_REGISTER_GLOBAL("tvm.contrib.mkl.batch_matmul_iterative")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DLTensor* A = args[0];
      ICHECK(TypeMatch(A->dtype, kDLFloat, 32) || TypeMatch(A->dtype, kDLFloat, 64));
      if (TypeMatch(A->dtype, kDLFloat, 32)) {
        CallBatchGemm(args, ret, MKLSgemmBatchIterativeOp());
      } else {
        CallBatchGemm(args, ret, MKLDgemmBatchIterativeOp());
      }
    });
}  // namespace contrib
}  // namespace tvm
