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

#include <cuda_fp16.h>
#include <float.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include "fp16_group_gemm.cuh"
#include "fp16_group_gemm_runner_sm100.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace tvm {
namespace runtime {

template <typename ElementA, typename ElementB, typename ElementC>
struct CutlassGroupGemm<100, ElementA, ElementB, ElementC> {
  static void run(ElementA* A, ElementB* B, int64_t* indptr, uint8_t* workspace, int workspace_size,
                  int N, int K, int num_groups, float alpha, float beta, ElementC* C,
                  cudaStream_t stream) {
    cutlass_group_gemm_sm100<ElementA, ElementB, ElementC>(
        A, B, indptr, workspace, workspace_size, N, K, num_groups, alpha, beta, C, stream);
  }
};

void tvm_cutlass_group_gemm_sm100(NDArray x, NDArray weight, NDArray indptr, NDArray workspace,
                                  NDArray out) {
  tvm_cutlass_group_gemm_impl<100>(x, weight, indptr, workspace, out);
}

TVM_FFI_REGISTER_GLOBAL("cutlass.group_gemm").set_body_typed(tvm_cutlass_group_gemm_sm100);

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
