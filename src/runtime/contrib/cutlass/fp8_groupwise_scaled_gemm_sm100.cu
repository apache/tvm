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

#include "../cublas/cublas_utils.h"
#include "fp8_groupwise_scaled_gemm.cuh"
#include "fp8_groupwise_scaled_gemm_runner_sm100.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace tvm {
namespace runtime {

template <typename TileShape, typename ClusterShape, typename ElementA, typename ElementB,
          typename ElementC, typename ElementBlockScale>
struct CutlassFP8GroupwiseGemm<100, TileShape, ClusterShape, ElementA, ElementB, ElementC,
                               ElementBlockScale> {
  static void run(ElementA* a, ElementB* b, ElementBlockScale* scales_a,
                  ElementBlockScale* scales_b, ElementC* out, uint8_t* workspace,
                  int64_t workspace_size, int64_t m, int64_t n, int64_t k, int64_t l,
                  cudaStream_t stream) {
    cutlass_fp8_groupwise_scaled_mm_sm100<TileShape, ClusterShape, ElementA, ElementB, ElementC,
                                          ElementBlockScale>(
        a, b, scales_a, scales_b, out, workspace, workspace_size, m, n, k, l, stream);
  }
};

void tvm_cutlass_fp8_groupwise_scaled_gemm_sm100(NDArray a, NDArray b, NDArray scales_a,
                                                 NDArray scales_b, NDArray workspace,
                                                 int64_t block_size_0, int64_t block_size_1,
                                                 NDArray out) {
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  tvm_cutlass_fp8_groupwise_scaled_gemm_impl<100, TileShape, ClusterShape>(
      a, b, scales_a, scales_b, workspace, block_size_0, block_size_1, out);
}

void tvm_cutlass_fp8_groupwise_scaled_bmm_sm100(NDArray a, NDArray b, NDArray scales_a,
                                                NDArray scales_b, NDArray workspace,
                                                int64_t block_size_0, int64_t block_size_1,
                                                NDArray out) {
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  tvm_cutlass_fp8_groupwise_scaled_bmm_impl<100, TileShape, ClusterShape>(
      a, b, scales_a, scales_b, workspace, block_size_0, block_size_1, out);
}

TVM_FFI_REGISTER_GLOBAL("cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn")
    .set_body_typed(tvm_cutlass_fp8_groupwise_scaled_gemm_sm100);
TVM_FFI_REGISTER_GLOBAL("cutlass.groupwise_scaled_bmm_e4m3fn_e4m3fn")
    .set_body_typed(tvm_cutlass_fp8_groupwise_scaled_bmm_sm100);

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
