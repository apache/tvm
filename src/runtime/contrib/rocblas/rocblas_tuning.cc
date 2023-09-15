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

int TuneRocblas(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue) {
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

    auto best_time = std::numeric_limits<double>::max();
    auto best_sol = 0;
    auto algo = rocblas_gemm_algo_solution_index;
    rocblas_int solutionIndex = 0;
    auto flags = rocblas_gemm_flags_none;
    int numRepeats = 10;

    // Get all solutions
    rocblas_int n_solutions;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, 
                                                        A_data, ab_type, lda, &beta, C_data, c_type, ldc,
                                                        C_data, c_type, ldc, 
                                                        compute_type,
                                                        algo, rocblas_gemm_flags_none, NULL,
                                                        &n_solutions));

    std::vector<rocblas_int> solutions(n_solutions);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_ex_get_solutions(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, 
                                                        A_data, ab_type, lda, &beta, C_data, c_type, ldc,
                                                        C_data, c_type, ldc, 
                                                        compute_type,
                                                        algo, rocblas_gemm_flags_none,
                                                        solutions.data(), &n_solutions));
    for (auto sol : solutions) {
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, 
                                            A_data, ab_type, lda, &beta, C_data, c_type, ldc,
                                            C_data, c_type, ldc, 
                                            compute_type,
                                            algo, sol, flags));
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(hdl, &stream));
        auto start = std::chrono::steady_clock::now();
        
        // timing loop
        for (rocblas_int c = 0; c < numRepeats; ++c) {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, 
                                                A_data, ab_type, lda, &beta, C_data, c_type, ldc,
                                                C_data, c_type, ldc, 
                                                compute_type,
                                                algo, sol, flags));
            hipDeviceSynchronize();
        }

        auto end = std::chrono::steady_clock::now();

        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();

        double avg_time = numRepeats ? (time / numRepeats) : 0;
        if (avg_time < best_time) {
            best_sol = sol;
            best_time = avg_time;
        }
    }
    solutionIndex = best_sol;
    return solutionIndex;
}


int TuneRocblasBatch(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue) {
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
    
    auto best_time = std::numeric_limits<double>::max();
    auto best_sol = 0;
    auto algo = rocblas_gemm_algo_solution_index;
    rocblas_int solutionIndex = 0;
    auto flags = rocblas_gemm_flags_none;
    int numRepeats = 20;

    // Get all solutions
    rocblas_int n_solutions;
    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, batch_stride_B,
                                                        A_data, ab_type, lda, batch_stride_A, &beta, C_data, c_type, ldc, batch_stride_C,
                                                        C_data, c_type, ldc, batch_stride_C, cnt,
                                                        compute_type,
                                                        algo, rocblas_gemm_flags_none, NULL,
                                                        &n_solutions));

    std::vector<rocblas_int> solutions(n_solutions);
    CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex_get_solutions(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, batch_stride_B,
                                                        A_data, ab_type, lda, batch_stride_A, &beta, C_data, c_type, ldc, batch_stride_C,
                                                        C_data, c_type, ldc, batch_stride_C, cnt,
                                                        compute_type,
                                                        algo, rocblas_gemm_flags_none,
                                                        solutions.data(), &n_solutions));
    for (auto sol : solutions) {
        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, batch_stride_B,
                                            A_data, ab_type, lda, batch_stride_A, &beta, C_data, c_type, ldc, batch_stride_C,
                                            C_data, c_type, ldc, batch_stride_C, cnt,
                                            compute_type,
                                            algo, sol, flags));
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(hdl, &stream));
        auto start = std::chrono::steady_clock::now();
        
        // timing loop
        for (rocblas_int c = 0; c < numRepeats; ++c) {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(hdl, trans_B, trans_A, N, M, K, &alpha, B_data, ab_type, ldb, batch_stride_B,
                                                A_data, ab_type, lda, batch_stride_A, &beta, C_data, c_type, ldc, batch_stride_C,
                                                C_data, c_type, ldc, batch_stride_C, cnt,
                                                compute_type,
                                                algo, sol, flags));
            hipDeviceSynchronize();
        }

        auto end = std::chrono::steady_clock::now();

        auto time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();

        double avg_time = numRepeats ? (time / numRepeats) : 0;
        if (avg_time < best_time) {
            best_sol = sol;
            best_time = avg_time;
        }
    }
    solutionIndex = best_sol;
    return solutionIndex;
}

}  // namespace contrib
}  // namespace tvm
