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
#include <tvm/ffi/function.h>

#include "fp16_group_gemm.cuh"
#include "fp16_group_gemm_runner_sm90.cuh"

namespace tvm {
namespace runtime {

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <typename ElementA, typename ElementB, typename ElementC>
struct CutlassGroupGemm<90, ElementA, ElementB, ElementC> {
  static void run(ElementA* A, ElementB* B, int64_t* indptr, uint8_t* workspace, int workspace_size,
                  int N, int K, int num_groups, float alpha, float beta, ElementC* C,
                  cudaStream_t stream) {
    cutlass_group_gemm_sm90<ElementA, ElementB, ElementC>(A, B, indptr, workspace, workspace_size,
                                                          N, K, num_groups, alpha, beta, C, stream);
  }
};

template <>
struct KernelTraits<cutlass::half_t> {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _256, _64>;  // Threadblock-level tile size
  using ClusterShape = Shape<_2, _2, _1>;    // Shape of the threadblocks in a cluster
};

template <>
struct KernelTraits<cutlass::bfloat16_t> {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _256, _64>;  // Threadblock-level tile size
  using ClusterShape = Shape<_2, _2, _1>;    // Shape of the threadblocks in a cluster
};

void tvm_cutlass_group_gemm_sm90(NDArray x, NDArray weight, NDArray indptr, NDArray workspace,
                                 NDArray out) {
  tvm_cutlass_group_gemm_impl<90>(x, weight, indptr, workspace, out);
}

TVM_FFI_REGISTER_GLOBAL("cutlass.group_gemm").set_body_typed(tvm_cutlass_group_gemm_sm90);

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

}  // namespace runtime
}  // namespace tvm
