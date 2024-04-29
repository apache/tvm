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
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "group_gemm_runner.cuh"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <>
struct KernelTraits<cutlass::half_t> {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _256, _64>;  // Threadblock-level tile size
  using ClusterShape = Shape<_2, _2, _1>;    // Shape of the threadblocks in a cluster
};

namespace tvm {
namespace runtime {

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_group_gemm_sm90(NDArray x, NDArray weight, NDArray indptr, NDArray workspace,
                                 NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  CHECK_EQ(x->ndim, 2);
  CHECK_EQ(weight->ndim, 3);
  CHECK_EQ(indptr->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 2);
  int num_groups = weight->shape[0];
  int n = weight->shape[1];
  int k = weight->shape[2];
  float alpha = 1.0f;
  float beta = 0.0f;
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
  cutlass_group_gemm(static_cast<ElementA*>(x->data), static_cast<ElementB*>(weight->data),
                     static_cast<int64_t*>(indptr->data), static_cast<uint8_t*>(workspace->data),
                     workspace->shape[0], n, k, num_groups, alpha, beta,
                     static_cast<ElementC*>(out->data), stream);
}

TVM_REGISTER_GLOBAL("cutlass.group_gemm_fp16_sm90")
    .set_body_typed(tvm_cutlass_group_gemm_sm90<cutlass::half_t, cutlass::half_t, cutlass::half_t>);

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
