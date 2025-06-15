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

#include "fp8_groupwise_scaled_group_gemm_runner_sm100.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace tvm {
namespace runtime {

void tvm_fp8_groupwise_scaled_group_gemm_sm100(NDArray a, NDArray b, NDArray scales_a,
                                               NDArray scales_b, NDArray indptr, NDArray workspace,
                                               int64_t block_size_0, int64_t block_size_1,
                                               NDArray out) {
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommended size is 4MB.
  static auto func = tvm::ffi::Function::GetGlobalRequired("runtime.get_cuda_stream");
  cudaStream_t stream = static_cast<cudaStream_t>(func().cast<void*>());
  CHECK_EQ(a->ndim, 2);
  CHECK_EQ(b->ndim, 3);
  CHECK_EQ(indptr->ndim, 1);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 2);
  int num_groups = b->shape[0];
  int n = b->shape[1];
  int k = b->shape[2];

  CHECK_EQ(scales_a->ndim, a->ndim);
  CHECK_EQ(scales_b->ndim, b->ndim);
  // scales_a is row-major of (m, k / block_size)
  CHECK_EQ((k + block_size_1 - 1) / block_size_1, scales_a->shape[1]);
  CHECK_EQ(scales_a->shape[0], a->shape[0]);
  // scales_b is col-major of (k / block_size, n / block_size)
  CHECK_EQ(scales_b->shape[0], num_groups);
  CHECK_EQ((n + block_size_0 - 1) / block_size_0, scales_b->shape[1]);
  CHECK_EQ((k + block_size_1 - 1) / block_size_1, scales_b->shape[2]);

  using tvm::runtime::DataType;
  CHECK_EQ(DataType(a->dtype), DataType::Float8E4M3FN());
  CHECK_EQ(DataType(b->dtype), DataType::Float8E4M3FN());
  CHECK_EQ(DataType(scales_a->dtype), DataType::Float(32));
  CHECK_EQ(DataType(scales_b->dtype), DataType::Float(32));
  CHECK_EQ(DataType(indptr->dtype), DataType::Int(64));
  CHECK_EQ(DataType(workspace->dtype), DataType::UInt(8));

  if (DataType(out->dtype) == DataType::Float(16)) {
    using Dtype = cutlass::half_t;
    cutlass_fp8_groupwise_scaled_group_gemm_sm100<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                  Dtype, float>(
        static_cast<cutlass::float_e4m3_t*>(a->data), static_cast<cutlass::float_e4m3_t*>(b->data),
        static_cast<float*>(scales_a->data), static_cast<float*>(scales_b->data),
        static_cast<int64_t*>(indptr->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0], n, k, num_groups, static_cast<Dtype*>(out->data), stream);
  } else if (DataType(out->dtype) == DataType::BFloat(16)) {
    using Dtype = cutlass::bfloat16_t;
    cutlass_fp8_groupwise_scaled_group_gemm_sm100<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                  Dtype, float>(
        static_cast<cutlass::float_e4m3_t*>(a->data), static_cast<cutlass::float_e4m3_t*>(b->data),
        static_cast<float*>(scales_a->data), static_cast<float*>(scales_b->data),
        static_cast<int64_t*>(indptr->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0], n, k, num_groups, static_cast<Dtype*>(out->data), stream);
  }
}

TVM_FFI_REGISTER_GLOBAL("cutlass.groupwise_scaled_group_gemm_e4m3fn_e4m3fn")
    .set_body_typed(tvm_fp8_groupwise_scaled_group_gemm_sm100);

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
