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
#include <tvm/ffi/function.h>
#include <tvm/ffi/function.h>

#include "../cublas/cublas_utils.h"
#include "blockwise_scaled_gemm_runner.cuh"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

namespace tvm {
namespace runtime {

void tvm_cutlass_fp8_blockwise_scaled_gemm(NDArray a, NDArray b, NDArray scales_a, NDArray scales_b,
                                           NDArray workspace, int64_t block_size_0,
                                           int64_t block_size_1, NDArray out) {
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  const auto get_stream_func = tvm::ffi::Function::GetGlobal("runtime.get_cuda_stream");
  ICHECK(get_stream_func.has_value());
  cudaStream_t stream = static_cast<cudaStream_t>((*get_stream_func)().cast<void*>());

  CHECK_GE(a->ndim, 2);
  CHECK_EQ(scales_a->ndim, a->ndim);
  CHECK_EQ(b->ndim, 2);
  CHECK_EQ(scales_b->ndim, 2);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, a->ndim);
  int64_t m = 1;
  for (int64_t i = 0; i < a->ndim - 1; ++i) {
    m *= a->shape[i];
  }
  int64_t n = b->shape[0];
  CHECK_EQ(a->shape[a->ndim - 1], b->shape[1]) << "Only col-major B is supported now.";
  int64_t k = a->shape[a->ndim - 1];

  // scales_a is col-major of (*a_shape[:-1], k / block_size)
  CHECK_EQ(scales_a->shape[0] * block_size_1, k);
  for (int64_t i = 1; i < scales_a->ndim; ++i) {
    CHECK_EQ(scales_a->shape[i], a->shape[i - 1]);
  }
  // scales_b is col-major of (k / block_size, n / block_size)
  CHECK_EQ(scales_b->shape[0] * block_size_0, n);
  CHECK_EQ(scales_b->shape[1] * block_size_1, k);

  using tvm::runtime::DataType;
  CHECK_EQ(DataType(a->dtype), DataType::NVFloat8E4M3());
  CHECK_EQ(DataType(b->dtype), DataType::NVFloat8E4M3());
  CHECK_EQ(DataType(scales_a->dtype), DataType::Float(32));
  CHECK_EQ(DataType(scales_b->dtype), DataType::Float(32));
  CHECK_EQ(DataType(workspace->dtype), DataType::UInt(8));

  if (DataType(out->dtype) == DataType::Float(16)) {
    cutlass_fp8_blockwise_scaled_gemm<TileShape, ClusterShape, cutlass::float_e4m3_t,
                                      cutlass::float_e4m3_t, cutlass::half_t, float>(
        static_cast<cutlass::float_e4m3_t*>(a->data), static_cast<cutlass::float_e4m3_t*>(b->data),
        static_cast<float*>(scales_a->data), static_cast<float*>(scales_b->data),
        static_cast<cutlass::half_t*>(out->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0] * DataType(workspace->dtype).bytes(), m, n, k, stream);
  } else if (DataType(out->dtype) == DataType::BFloat(16)) {
    cutlass_fp8_blockwise_scaled_gemm<TileShape, ClusterShape, cutlass::float_e4m3_t,
                                      cutlass::float_e4m3_t, cutlass::bfloat16_t, float>(
        static_cast<cutlass::float_e4m3_t*>(a->data), static_cast<cutlass::float_e4m3_t*>(b->data),
        static_cast<float*>(scales_a->data), static_cast<float*>(scales_b->data),
        static_cast<cutlass::bfloat16_t*>(out->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0] * DataType(workspace->dtype).bytes(), m, n, k, stream);
  } else {
    LOG(FATAL) << "Unsupported output dtype: " << DataType(out->dtype);
  }
}

void tvm_cutlass_fp8_blockwise_scaled_bmm(NDArray a, NDArray b, NDArray scales_a, NDArray scales_b,
                                          NDArray workspace, int64_t block_size_0,
                                          int64_t block_size_1, NDArray out) {
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  const auto get_stream_func = tvm::ffi::Function::GetGlobal("runtime.get_cuda_stream");
  ICHECK(get_stream_func.has_value());
  cudaStream_t stream = static_cast<cudaStream_t>((*get_stream_func)().cast<void*>());

  CHECK_EQ(a->ndim, 3);
  CHECK_EQ(scales_a->ndim, 3);
  CHECK_EQ(b->ndim, 3);
  CHECK_EQ(scales_b->ndim, 3);
  CHECK_EQ(workspace->ndim, 1);
  CHECK_EQ(out->ndim, 3);
  int64_t batch_size = a->shape[0];
  int64_t m = a->shape[1];
  int64_t n = b->shape[1];
  CHECK_EQ(a->shape[2], b->shape[2]) << "Only col-major B is supported now.";
  int64_t k = a->shape[2];
  CHECK_EQ(b->shape[0], batch_size);
  CHECK_EQ(scales_a->shape[0], batch_size);
  CHECK_EQ(scales_b->shape[0], batch_size);
  CHECK_EQ(out->shape[0], batch_size);

  // scales_a is col-major of (batch_size, m, k / block_size)
  CHECK_EQ(scales_a->shape[1] * block_size_1, k);
  CHECK_EQ(scales_a->shape[2], m);
  // scales_b is col-major of (k / block_size, n / block_size)
  CHECK_EQ(scales_b->shape[1] * block_size_0, n);
  CHECK_EQ(scales_b->shape[2] * block_size_1, k);

  using tvm::runtime::DataType;
  CHECK_EQ(DataType(a->dtype), DataType::NVFloat8E4M3());
  CHECK_EQ(DataType(b->dtype), DataType::NVFloat8E4M3());
  CHECK_EQ(DataType(scales_a->dtype), DataType::Float(32));
  CHECK_EQ(DataType(scales_b->dtype), DataType::Float(32));
  CHECK_EQ(DataType(workspace->dtype), DataType::UInt(8));

  if (DataType(out->dtype) == DataType::Float(16)) {
    cutlass_fp8_blockwise_scaled_bmm<TileShape, ClusterShape, cutlass::float_e4m3_t,
                                     cutlass::float_e4m3_t, cutlass::half_t, float>(
        static_cast<cutlass::float_e4m3_t*>(a->data), static_cast<cutlass::float_e4m3_t*>(b->data),
        static_cast<float*>(scales_a->data), static_cast<float*>(scales_b->data),
        static_cast<cutlass::half_t*>(out->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0] * DataType(workspace->dtype).bytes(), m, n, k, batch_size, stream);
  } else if (DataType(out->dtype) == DataType::BFloat(16)) {
    cutlass_fp8_blockwise_scaled_bmm<TileShape, ClusterShape, cutlass::float_e4m3_t,
                                     cutlass::float_e4m3_t, cutlass::bfloat16_t, float>(
        static_cast<cutlass::float_e4m3_t*>(a->data), static_cast<cutlass::float_e4m3_t*>(b->data),
        static_cast<float*>(scales_a->data), static_cast<float*>(scales_b->data),
        static_cast<cutlass::bfloat16_t*>(out->data), static_cast<uint8_t*>(workspace->data),
        workspace->shape[0] * DataType(workspace->dtype).bytes(), m, n, k, batch_size, stream);
  } else {
    LOG(FATAL) << "Unsupported output dtype: " << DataType(out->dtype);
  }
}

TVM_FFI_REGISTER_GLOBAL("cutlass.blockwise_scaled_gemm_e4m3fn_e4m3fn")
    .set_body_typed(tvm_cutlass_fp8_blockwise_scaled_gemm);
TVM_FFI_REGISTER_GLOBAL("cutlass.blockwise_scaled_bmm_e4m3fn_e4m3fn")
    .set_body_typed(tvm_cutlass_fp8_blockwise_scaled_bmm);

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
