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

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

namespace tvm {
namespace runtime {

template <int Arch, typename ElementA, typename ElementB, typename ElementC>
struct CutlassGroupGemm;

template <int Arch>
void tvm_cutlass_group_gemm_impl(NDArray x, NDArray weight, NDArray indptr, NDArray workspace,
                                 NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  static auto func = tvm::ffi::Function::GetGlobalRequired("runtime.get_cuda_stream");
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
  cudaStream_t stream = static_cast<cudaStream_t>(func().cast<void*>());

  if (DataType(x->dtype) == DataType::Float(16)) {
    CHECK(DataType(weight->dtype) == DataType::Float(16));
    CHECK(DataType(out->dtype) == DataType::Float(16));
    using Dtype = cutlass::half_t;
    CutlassGroupGemm<Arch, Dtype, Dtype, Dtype>::run(
        static_cast<Dtype*>(x->data), static_cast<Dtype*>(weight->data),
        static_cast<int64_t*>(indptr->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0], n, k, num_groups, alpha, beta, static_cast<Dtype*>(out->data), stream);
  } else if (DataType(x->dtype) == DataType::BFloat(16)) {
    CHECK(DataType(weight->dtype) == DataType::BFloat(16));
    CHECK(DataType(out->dtype) == DataType::BFloat(16));
    using Dtype = cutlass::bfloat16_t;
    CutlassGroupGemm<Arch, Dtype, Dtype, Dtype>::run(
        static_cast<Dtype*>(x->data), static_cast<Dtype*>(weight->data),
        static_cast<int64_t*>(indptr->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0], n, k, num_groups, alpha, beta, static_cast<Dtype*>(out->data), stream);
  }
}

}  // namespace runtime
}  // namespace tvm
