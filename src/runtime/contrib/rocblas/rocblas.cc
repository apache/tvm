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

#include <dmlc/thread_local.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include "../../3rdparty/compiler-rt/builtin_fp16.h"
#include "../cblas/gemm_common.h"
#include "rocblas_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

void CallHipblasLt(hipblasLtHandle_t hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue) {
  ICHECK(TypeEqual(A->dtype, B->dtype));
  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;
  

  auto compute_type = HIPBLASLT_COMPUTE_F32;
  auto scale_type = HIPBLASLT_R_32F;
  auto ab_type = HIPBLASLT_R_32F;
  auto c_type = HIPBLASLT_R_32F;
  float one_fp32 = 1.0;
  float zero_fp32 = 0.0;
  void* alpha = &one_fp32;
  void* beta = &zero_fp32;

  uint64_t max_workspace_size = 32 * 1024 * 1024;
  void* d_workspace = nullptr;

  if (A->dtype.bits == 16 && A->dtype.code == kDLFloat) {
    ab_type = HIPBLASLT_R_16F;
  }

  if (C->dtype.bits == 16 && C->dtype.code == kDLFloat) {
    c_type = HIPBLASLT_R_16F;
    compute_type = HIPBLASLT_COMPUTE_F32;
    scale_type = HIPBLASLT_R_32F;
  }

  hipblasLtMatmulDesc_t op_desc;
  hipblasOperation_t  op_transa = transa ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t  op_transb = transb ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &op_transb, sizeof(op_transa)));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &op_transa, sizeof(op_transb)));

  if (bias != nullptr) {
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                      &bias->data, sizeof(float*)));
  }

  if (epilogue != HIPBLASLT_EPILOGUE_DEFAULT) {
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                      &epilogue, sizeof(epilogue)));
  }

  int batch_offset_A = A->ndim - 2;
  int batch_offset_B = B->ndim - 2;

  int M = ColumnCount(B, transb, batch_offset_B);
  int N = RowCount(A, transa, batch_offset_A);
  int K = ColumnCount(A, transa, batch_offset_A);
  bool use_batched_gemm = A->ndim > 2 || B->ndim > 2;

  // If A is batched but B is not, flatten all non-reduction axes of A to use the regular GEMM.
  // This trick is only applicable if batch axes and the other spatial axis (M or N) are
  // adjacent in both the input and the output matrix. In particular, if A is of shape (M, K)
  // and B matrix is of shape (Batch, N, K) with transb = true, the output shape
  // is (Batch, M, N). Since the Batch and the N axes are not adjacent in the output, we cannot
  // use the regular GEMM if only B is batched.
  if (A->ndim > 2 && B->ndim == 2 && transa == false) {
    N = 1;
    for (int i = 0; i < A->ndim - 1; ++i) {
      N *= A->shape[i];
    }
    use_batched_gemm = false;
  }

  int lda = transb ? K : M;
  int ldb = transa ? N : K;
  int ldc = M;

  hipblasLtMatrixLayout_t A_desc, B_desc, C_desc;
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&A_desc, ab_type, !transb ? M : K, !transb ? K : M, lda));
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatrixLayoutCreate(&B_desc, ab_type, !transa ? K : N, !transa ? N : K, ldb));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&C_desc, c_type, M, N, ldc));

  if (use_batched_gemm) {
    auto get_batch_count = [](int64_t* shape, int batch_offset) {
      int64_t count = 1;
      for (int i = 0; i < batch_offset; ++i) {
        count *= shape[i];
      }
      return count;
    };
    auto set_batch = [](hipblasLtMatrixLayout_t mat_desc, int batch_count, int64_t batch_stride) {
      CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutSetAttribute(
          mat_desc, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
      CHECK_HIPBLASLT_ERROR(
          hipblasLtMatrixLayoutSetAttribute(mat_desc, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                           &batch_stride, sizeof(batch_stride)));
    };

    int batch_count_A = get_batch_count(A->shape, batch_offset_A);
    int batch_count_B = get_batch_count(B->shape, batch_offset_B);
    int batch_count_C = get_batch_count(C->shape, C->ndim - 2);
    int64_t batch_stride_A = M * K;
    int64_t batch_stride_B = K * N;
    int64_t batch_stride_C = M * N;

    // HipBLASLt does not seem to support batched GEMM with one of matrices having
    // one batch (with batch_stride 0).
    ICHECK_EQ(batch_count_A, batch_count_B);

    set_batch(A_desc, batch_count_A, batch_stride_A);
    set_batch(B_desc, batch_count_B, batch_stride_B);
    set_batch(C_desc, batch_count_C, batch_stride_C);
  }

  auto A_data = static_cast<char*>(A->data) + A->byte_offset;
  auto B_data = static_cast<char*>(B->data) + B->byte_offset;
  auto C_data = static_cast<char*>(C->data) + C->byte_offset;

  // Set User Preference attributes
  hipblasLtMatmulPreference_t pref;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
  CHECK_HIPBLASLT_ERROR(
      hipblasLtMatmulPreferenceSetAttribute(pref,
                                            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &max_workspace_size,
                                            sizeof(max_workspace_size)));

  const int                        request_solutions = 1;
  hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
  int                              returnedAlgoCount = 0;
  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(hdl,
                                                        op_desc,
                                                        A_desc,
                                                        B_desc,
                                                        C_desc,
                                                        C_desc,
                                                        pref,
                                                        request_solutions,
                                                        heuristicResult,
                                                        &returnedAlgoCount));
  
  uint64_t workspace_size = 0;
  workspace_size = std::max(workspace_size, heuristicResult[0].workspaceSize);
  if(workspace_size > 0) {
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));
  }

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(hdl, op_desc, alpha, B_data, A_desc, A_data, B_desc, beta,
                                    C_data, C_desc, C_data, C_desc, &heuristicResult[0].algo, d_workspace, workspace_size, stream));

  CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(op_desc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(A_desc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(B_desc));
  CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(C_desc));
  CHECK_HIP_ERROR(hipFree(d_workspace));
}


void CallRocblas(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue, size_t algo) {
  ICHECK(TypeEqual(A->dtype, B->dtype));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  auto compute_type = rocblas_datatype_f32_r;
  auto ab_type = rocblas_datatype_f32_r;
  auto c_type = rocblas_datatype_f32_r;
  float one_fp32 = 1.0;
  float zero_fp32 = 0.0;
  float alpha = 1.0;
  float beta = 0.0;

  if (A->dtype.bits == 16 && A->dtype.code == kDLFloat) {
    ab_type = rocblas_datatype_f16_r;
  }

  if (C->dtype.bits == 16 && C->dtype.code == kDLFloat) {
    c_type = rocblas_datatype_f16_r;
    compute_type = rocblas_datatype_f32_r;
  }

  rocblas_operation trans_A = transa ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation trans_B = transb ? rocblas_operation_transpose : rocblas_operation_none;

  size_t N = transb ? B->shape[0] : B->shape[1];
  size_t M = transa ? A->shape[1] : A->shape[0];
  size_t K = transb ? B->shape[1] : B->shape[0];
  size_t lda = transa ? M : K;
  size_t ldb = transb ? K : N;
  size_t ldc = N;

  void* A_data = nullptr;
  void* B_data = nullptr;
  void* C_data = nullptr;

  A_data = reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset);
  B_data = reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset);
  C_data = reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset);

  if (A->dtype.bits == 16 && A->dtype.code == kDLFloat) {
    A_data = reinterpret_cast<rocblas_half*>(static_cast<char*>(A->data) + A->byte_offset);
    B_data = reinterpret_cast<rocblas_half*>(static_cast<char*>(B->data) + B->byte_offset);
  }
  if (C->dtype.bits == 16 && C->dtype.code == kDLFloat) {
    C_data = reinterpret_cast<rocblas_half*>(static_cast<char*>(C->data) + C->byte_offset);
  }

  CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, 
                                      A_data, ab_type, lda, &beta, C_data, c_type, ldc,
                                      C_data, c_type, ldc, 
                                      compute_type, rocblas_gemm_algo_solution_index, algo, rocblas_gemm_flags_none));
}

void CallRocblasBatch(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue, size_t algo_batch) {
  ICHECK(TypeEqual(A->dtype, B->dtype));

  // Reversed strides indicates an in-place transpose operation.
  transa = IsInPlaceTransposed(A) ? !transa : transa;
  transb = IsInPlaceTransposed(B) ? !transb : transb;

  auto compute_type = rocblas_datatype_f32_r;
  auto ab_type = rocblas_datatype_f32_r;
  auto c_type = rocblas_datatype_f32_r;
  float alpha = 1.0;
  float beta = 0.0;

  if (A->dtype.bits == 16 && A->dtype.code == kDLFloat) {
    ab_type = rocblas_datatype_f16_r;
  }

  if (C->dtype.bits == 16 && C->dtype.code == kDLFloat) {
    c_type = rocblas_datatype_f16_r;
    compute_type = rocblas_datatype_f32_r;
  }

  rocblas_operation trans_A = transa ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation trans_B = transb ? rocblas_operation_transpose : rocblas_operation_none;

  int batch_offset_A = A->ndim - 2;
  int batch_offset_B = B->ndim - 2;
  bool use_batched_gemm = true;
  
  size_t N = transb ? B->shape[batch_offset_B] : B->shape[batch_offset_B + 1];
  size_t M = transa ? A->shape[batch_offset_A + 1] : A->shape[batch_offset_A];
  size_t K = transb ? B->shape[batch_offset_B + 1] : B->shape[batch_offset_B];

  if (A->ndim > 2 && B->ndim == 2 && transa == false) {
    ICHECK(A->ndim==3 && A->shape[0]==1) << "Not support now";
  }

  size_t lda = transa ? M : K;
  size_t ldb = transb ? K : N;
  size_t ldc = N;

  auto get_batch_count = [](int64_t* shape, int batch_offset) {
    int64_t count = 1;
    for (int i = 0; i < batch_offset; ++i) {
      count *= shape[i];
    }
    return count;
  };
  int batch_count_A = get_batch_count(A->shape, batch_offset_A);
  int batch_count_B = get_batch_count(B->shape, batch_offset_B);
  int batch_count_C = get_batch_count(C->shape, C->ndim - 2);
  ICHECK_EQ(batch_count_A, batch_count_B);
  ICHECK_EQ(batch_count_A, batch_count_C);
  rocblas_int cnt = std::max(batch_count_A, 1);

  int64_t batch_stride_A = M * K;
  int64_t batch_stride_B = K * N;
  int64_t batch_stride_C = M * N;


  void* A_data = nullptr;
  void* B_data = nullptr;
  void* C_data = nullptr;

  A_data = reinterpret_cast<float*>(static_cast<char*>(A->data) + A->byte_offset);
  B_data = reinterpret_cast<float*>(static_cast<char*>(B->data) + B->byte_offset);
  C_data = reinterpret_cast<float*>(static_cast<char*>(C->data) + C->byte_offset);

  if (A->dtype.bits == 16 && A->dtype.code == kDLFloat) {
    A_data = reinterpret_cast<rocblas_half*>(static_cast<char*>(A->data) + A->byte_offset);
    B_data = reinterpret_cast<rocblas_half*>(static_cast<char*>(B->data) + B->byte_offset);
  }
  if (C->dtype.bits == 16 && C->dtype.code == kDLFloat) {
    C_data = reinterpret_cast<rocblas_half*>(static_cast<char*>(C->data) + C->byte_offset);
  }

  CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, batch_stride_B,
                                      A_data, ab_type, lda, batch_stride_A, &beta, C_data, c_type, ldc, batch_stride_C,
                                      C_data, c_type, ldc, batch_stride_C, cnt,
                                      compute_type, rocblas_gemm_algo_solution_index, algo_batch, rocblas_gemm_flags_none));
}

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
