# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-outside-toplevel, invalid-name
"""Instantiate a C++ source for profiling CUTLASS kernels."""


class GemmProfilerEmitter(object):
    """Emit a C++ source for profiling CUTLASS kernels."""

    def __init__(self):
        from jinja2 import Template

        self.template = Template(
            """
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "cutlass/gemm/device/gemm.h"

#define CUTLASS_CHECK(status)                                                                    \\
  {                                                                                              \\
    cutlass::Status error = status;                                                              \\
    if (error != cutlass::Status::kSuccess) {                                                    \\
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \\
                << std::endl;                                                                    \\
      exit(EXIT_FAILURE);                                                                        \\
    }                                                                                            \\
  }

#define CUDA_CHECK(status)                                              \\
  {                                                                     \\
    cudaError_t error = status;                                         \\
    if (error != cudaSuccess) {                                         \\
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \\
                << " at line: " << __LINE__ << std::endl;               \\
      exit(EXIT_FAILURE);                                               \\
    }                                                                   \\
  }

template<typename DTypeA, typename DTypeB, typename DTypeC>
cudaError_t CutlassGemm(
    int M,
    int N,
    int K,
    DTypeC alpha,
    DTypeA const *A,
    int lda,
    DTypeB const *B,
    int ldb,
    DTypeC beta,
    DTypeC *C,
    int ldc) {
  using namespace std::chrono;
  {{OperatorDef}}
  Operation_{{OperatorName}} gemm_operator;
  Operation_{{OperatorName}}::Arguments args({M, N, K},
                              {A, lda},
                              {B, ldb},
                              {C, ldc},
                              {C, ldc},
                              {alpha, beta});
  cutlass::Status status = gemm_operator(args);
  CUTLASS_CHECK(status)

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) {
    status = gemm_operator(args);
  }
  cudaDeviceSynchronize();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << time_span.count() << std::endl;
  return cudaSuccess;
}


template<typename DType>
cudaError_t AllocateMatrix(DType **matrix, int ldm, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(DType) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

template<typename DTypeA, typename DTypeB, typename DTypeC>
cudaError_t TestCutlassGemm(int M, int N, int K, DTypeC alpha, DTypeC beta) {
  cudaError_t result;

  {{LeadingDim}}
  // size_t sizeof_C = sizeof(DTypeC) * ldc * N;
  DTypeA *A;
  DTypeB *B;
  DTypeC *C_cutlass;
  result = AllocateMatrix<DTypeA>(&A, lda, M, K, 0);
  if (result !=  cudaSuccess) {
    return result;
  }
  result = AllocateMatrix<DTypeB>(&B, ldb, K, N, 17);
  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }
  result = AllocateMatrix<DTypeC>(&C_cutlass, ldc, M, N, 101);
  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }
  result = CutlassGemm<DTypeA, DTypeB, DTypeC>(M, N, K, alpha, A, lda, B, ldb,
                                                  beta, C_cutlass, ldc);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);
  return cudaSuccess;
}

int main(int argc, const char *arg[]) {
  int problem[3] = { 4096, 4096, 4096 };
  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }
  float scalars[2] = { 1, 0 };
  cudaError_t result = TestCutlassGemm< {{DTypeA}}, {{DTypeB}}, {{DTypeC}}>(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    static_cast<{{DTypeC}}>(scalars[0]),     // alpha
    static_cast<{{DTypeC}}>(scalars[1])      // beta
  );
  return result == cudaSuccess ? 0 : -1;
}
"""
        )

    def emit(self, op_name, op_def, dtype_a, dtype_b, dtype_c, ld):
        src = self.template.render(
            OperatorName=op_name,
            OperatorDef=op_def,
            DTypeA=dtype_a,
            DTypeB=dtype_b,
            DTypeC=dtype_c,
            LeadingDim=ld,
        )
        return src
